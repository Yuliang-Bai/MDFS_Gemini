import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
from typing import Dict, List
from math import floor, log
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from ...base import BaseMethod, FitResult


def select_from_weights(weights_dict: Dict[str, np.ndarray], X_dict: Dict[str, np.ndarray], n_samples: int) -> Dict[
    str, List[int]]:
    if n_samples > 1:
        k_total = max(1, floor(n_samples / np.log(n_samples)))
    else:
        k_total = 1

    total_dim = sum([X_dict[name].shape[1] for name in X_dict.keys()])
    selected = {}

    for name, W in weights_dict.items():
        d_v = X_dict[name].shape[1]
        k_v = int(k_total * (d_v / total_dim)) if total_dim > 0 else 0

        if W.ndim > 1:
            scores = np.linalg.norm(W, axis=1)
        else:
            scores = np.abs(W)

        sorted_idx = np.argsort(scores)[::-1]
        top_indices = sorted_idx[:k_v]
        selected_list = top_indices.tolist()
        selected_list.sort()
        selected[name] = selected_list

    return selected


# ---------------------------------------------------------
# 1. SMVFS
# ---------------------------------------------------------
class SMVFS(BaseMethod):
    def __init__(self, name="SMVFS", alpha=0.1, rho=1.0, mu=1.0, max_iter=50, **kwargs):
        super().__init__(name, **kwargs)
        self.lam = kwargs.get('lambda_', alpha)
        self.rho = kwargs.get('rho', rho)
        self.mu_param = kwargs.get('mu', mu)
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        scaler = StandardScaler()
        X_std = {v: scaler.fit_transform(X[v]) for v in X}

        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        if Y.shape[1] == 1: Y = np.hstack([1 - Y, Y])

        n_samples, n_classes = Y.shape
        view_names = list(X.keys())
        n_views = len(view_names)

        W = {v: np.zeros((X_std[v].shape[1], n_classes)) for v in view_names}
        Z_bar = np.zeros((n_samples, n_classes))
        U = np.zeros((n_samples, n_classes))
        B = {v: np.zeros_like(W[v]) for v in view_names}
        V = {v: np.zeros_like(W[v]) for v in view_names}

        Inv_matrices = {}
        for v in view_names:
            XtX = X_std[v].T @ X_std[v]
            d_v = XtX.shape[0]
            tmp = self.rho * XtX + self.mu_param * np.eye(d_v)
            Inv_matrices[v] = np.linalg.inv(tmp)

        for k in range(self.max_iter):
            sum_XW = np.zeros_like(Z_bar)
            for v in view_names: sum_XW += X_std[v] @ W[v]
            mean_XW = sum_XW / n_views

            for v in view_names:
                A_v = Z_bar + (X_std[v] @ W[v]) - mean_XW - U
                term1 = self.rho * (X_std[v].T @ A_v)
                term2 = self.mu_param * (B[v] - V[v])
                W[v] = Inv_matrices[v] @ (term1 + term2)

                target_B = W[v] + V[v]
                thresh = self.lam / self.mu_param
                norms = np.linalg.norm(target_B, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                scale = np.maximum(0, 1 - thresh / norms)
                B[v] = target_B * scale
                V[v] = V[v] + W[v] - B[v]

            sum_XW_new = np.zeros_like(Z_bar)
            for v in view_names: sum_XW_new += X_std[v] @ W[v]
            mean_XW_new = sum_XW_new / n_views
            Z_bar = (Y + self.rho * mean_XW_new + self.rho * U) / (n_views + self.rho)
            U = U + mean_XW_new - Z_bar

        return FitResult(selected_features=select_from_weights(W, X_std, n_samples=n_samples))


# ---------------------------------------------------------
# 2. HLRFS (【极大优化提速版本】)
# ---------------------------------------------------------
class HLRFS(BaseMethod):
    def __init__(self, name="HLRFS", alpha=1.0, lambda_=0.1, r=5, n_neighbors=5, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.alpha = kwargs.get('beta', alpha)
        self.lam = kwargs.get('gamma', lambda_)
        self.r = r
        self.k = n_neighbors
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        scaler = StandardScaler()
        X_std_list = [scaler.fit_transform(v) for v in X.values()]
        X_concat = np.hstack(X_std_list)
        n, d = X_concat.shape

        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        if Y.shape[1] == 1: Y = np.hstack([1 - Y, Y])

        knn_graph = kneighbors_graph(X_concat, self.k, mode='connectivity', include_self=True)
        H_inc = knn_graph.toarray().T
        Dv = np.sum(H_inc, axis=1)
        De = np.sum(H_inc, axis=0)
        Dv_inv_half = np.diag(np.power(Dv, -0.5, where=Dv != 0))
        De_inv = np.diag(np.power(De, -1.0, where=De != 0))
        S_hat = Dv_inv_half @ H_inc @ De_inv @ H_inc.T @ Dv_inv_half
        L = np.eye(n) - S_hat

        H_c = np.eye(n) - np.ones((n, n)) / n
        Xc = H_c @ X_concat
        Yc = H_c @ Y
        XtYc = Xc.T @ Yc

        # 强制对称以防由于浮点数误差带来的错误
        Sb = XtYc @ XtYc.T
        Sb = 0.5 * (Sb + Sb.T)
        XLXt = X_concat.T @ (H_c @ L @ H_c) @ X_concat

        D_diag = np.eye(d)
        A = np.random.randn(d, self.r)

        for iter_ in range(self.max_iter):
            Sw = (Xc.T @ Xc) + (self.alpha * XLXt) + (self.lam * D_diag)
            Sw = 0.5 * (Sw + Sw.T)
            Sw += 1e-6 * np.eye(d)

            # 【核心加速策略】：由于 d 可能高达数千，只需计算特征值最大的 r 个特征向量即可
            try:
                vals, vecs = scipy.linalg.eigh(Sb, Sw, subset_by_index=(d - self.r, d - 1))
                A = vecs
            except Exception:
                try:
                    vals, vecs = eigsh(Sb, M=Sw, k=self.r, which='LA')
                    A = vecs
                except Exception:
                    vals, vecs = scipy.linalg.eigh(Sb + 1e-6 * np.eye(d))
                    A = vecs[:, -self.r:]

            term1 = np.linalg.inv(A.T @ Sw @ A + 1e-6 * np.eye(self.r))
            term2 = A.T @ XtYc
            B = term1 @ term2

            W = A @ B
            row_norms = np.linalg.norm(W, axis=1)
            d_vals = 1.0 / (2.0 * row_norms + 1e-10)
            D_diag = np.diag(d_vals)

        W = A @ B
        W_dict = {}
        curr = 0
        for v_name in X.keys():
            dim = X[v_name].shape[1]
            W_dict[v_name] = W[curr:curr + dim, :]
            curr += dim

        return FitResult(selected_features=select_from_weights(W_dict, X, n_samples=n))


# ---------------------------------------------------------
# 3. SCFS
# ---------------------------------------------------------
class SCFS(BaseMethod):
    def __init__(self, name="SCFS", lambda1=0.1, lambda2=0.1, rho=1.0, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.l1 = kwargs.get('lambda_', lambda1)
        self.l2 = lambda2
        self.rho = rho
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        scaler = StandardScaler()
        X_std = {v: scaler.fit_transform(X[v]) for v in X}
        view_names = sorted(X.keys())
        n_views = len(view_names)
        n_samples = len(y)

        lb = LabelBinarizer()
        Y_full = lb.fit_transform(y)
        if Y_full.shape[1] == 1: Y_full = np.hstack([1 - Y_full, Y_full])
        Y_full = Y_full.astype(float)

        classes = np.unique(y)
        n_classes = len(classes)

        X_indices = {idx: np.where(y == c)[0] for idx, c in enumerate(classes)}
        W = {v: np.zeros((X_std[v].shape[1], n_classes)) for v in view_names}
        T_bar = {i: np.zeros((len(rows), n_classes)) for i, rows in X_indices.items()}
        P_bar = {i: np.zeros((len(rows), n_classes)) for i, rows in X_indices.items()}

        XtX_cache = {i: {v: X_std[v][rows].T @ X_std[v][rows] for v in view_names} for i, rows in X_indices.items()}

        for k in range(self.max_iter):
            for v in view_names:
                d_j = X_std[v].shape[1]
                G_mat = np.diag(1.0 / (2.0 * np.linalg.norm(W[v], axis=1) + 1e-6))
                Sum_XtX = sum(XtX_cache[i][v] for i in range(n_classes))

                for c_idx in range(n_classes):
                    Q_mat = (1.0 / (2.0 * np.linalg.norm(W[v][:, c_idx]) + 1e-6)) * np.eye(d_j)
                    LHS = 2 * self.l1 * G_mat + self.rho * Sum_XtX + 2 * self.l2 * Q_mat
                    RHS = np.zeros(d_j)

                    for i in range(n_classes):
                        rows = X_indices[i]
                        X_ij = X_std[v][rows]
                        pred_sum = sum(X_std[v_temp][rows] @ W[v_temp] for v_temp in view_names)
                        pred_mean = pred_sum / n_views

                        residual = T_bar[i][:, c_idx] + X_ij @ W[v][:, c_idx] - pred_mean[:, c_idx] - (
                                    P_bar[i][:, c_idx] / self.rho)
                        RHS += self.rho * (X_ij.T @ residual)

                    W[v][:, c_idx] = np.linalg.solve(LHS + 1e-6 * np.eye(d_j), RHS)

            for i in range(n_classes):
                rows = X_indices[i]
                Y_i = Y_full[rows]
                pred_mean = sum(X_std[v][rows] @ W[v] for v in view_names) / n_views
                H_i = pred_mean + (P_bar[i] / self.rho) - (Y_i / n_views)
                h_norms = np.linalg.norm(H_i, axis=1, keepdims=True)
                h_norms[h_norms == 0] = 1e-10
                scale = np.maximum(0, 1 - (1.0 / (2.0 * self.rho)) / h_norms)
                T_bar[i] = H_i * scale + (Y_i / n_views)

            for i in range(n_classes):
                rows = X_indices[i]
                pred_mean = sum(X_std[v][rows] @ W[v] for v in view_names) / n_views
                P_bar[i] = P_bar[i] + self.rho * (pred_mean - T_bar[i])

        return FitResult(selected_features=select_from_weights(W, X_std, n_samples=n_samples))