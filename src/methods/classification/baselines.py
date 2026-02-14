import numpy as np
import scipy.linalg
from typing import Dict, List, Optional
from math import floor, log
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# 假设 BaseMethod 和 FitResult 已经在你的项目中定义
from ...base import BaseMethod, FitResult


# ---------------------------------------------------------
# 辅助函数 (已集成 N/logN 策略 + 排序保护)
# ---------------------------------------------------------

def select_from_weights(weights_dict: Dict[str, np.ndarray], X_dict: Dict[str, np.ndarray], n_samples: int) -> Dict[
    str, List[int]]:
    """
    【最终修正版 - 按维度比例分配】
    1. 总数量策略：遵循 N/logN 规则。
       k_total = floor(n_samples / log(n_samples))
    2. 分配策略：不再是全局竞争，而是按模态维度的比例分配名额 (Proportional Allocation)。
       k_v = k_total * (dim_v / total_dim)
    3. 筛选逻辑：在每个模态内部进行排序截断。
    """
    # 1. 计算总特征数配额 k_total
    if n_samples > 1:
        # 防止 log(1) = 0 的情况，且至少选 1 个
        k_total = max(1, floor(n_samples / np.log(n_samples)))
    else:
        k_total = 1

    # 2. 计算总维度
    # 注意：weights_dict[name] 的 shape[0] 即为该视图的特征维度 d_v
    total_dim = sum([X_dict[name].shape[1] for name in X_dict.keys()])

    selected = {}

    # 3. 遍历每个视图，按比例分配并筛选
    for name, W in weights_dict.items():
        # 当前视图维度
        d_v = X_dict[name].shape[1]

        # --- 关键修改：计算该视图的配额 k_v ---
        # 使用 round 或 int 均可，这里用 int 向下取整，最后不足的部分通常影响不大
        # 或者为了保证至少选 1 个（如果 k_total 允许），可以使用 max(1, ...)
        if total_dim > 0:
            ratio = d_v / total_dim
            k_v = int(k_total * ratio)
        else:
            k_v = 0

        # 边界保护：如果算出来是 0 但总配额很大，可能因维度极不平衡导致。
        # 这里严格遵循比例，如果是 0 就选 0（或者你可以改为 max(1, k_v)）
        if k_v == 0 and k_total > 0 and d_v > 0:
            # 可选：如果希望每个视图至少有一个机会，可以取消下面注释
            # k_v = 1
            pass

        # --- 计算特征分数 ---
        if W.ndim > 1:
            scores = np.linalg.norm(W, axis=1)  # L2,1 norm
        else:
            scores = np.abs(W)

        # --- 视图内部排序 ---
        # 降序排列的索引
        sorted_idx = np.argsort(scores)[::-1]

        # --- 截断前 k_v 个 ---
        # 即使 k_v 超过了 d_v (不可能发生)，切片操作也是安全的
        top_indices = sorted_idx[:k_v]

        # 存入结果并排序索引（美观）
        selected_list = top_indices.tolist()
        selected_list.sort()
        selected[name] = selected_list

    return selected


# ---------------------------------------------------------
# 1. SMVFS (Sharing Multi-view Feature Selection)
# ---------------------------------------------------------
class SMVFS(BaseMethod):
    def __init__(self, name="SMVFS", alpha=0.1, rho=1.0, mu=1.0, max_iter=50, **kwargs):
        super().__init__(name, **kwargs)
        self.lam = kwargs.get('lambda_', alpha)
        self.rho = kwargs.get('rho', rho)
        self.mu_param = kwargs.get('mu', mu)
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # 1. 数据标准化
        scaler = StandardScaler()
        X_std = {v: scaler.fit_transform(X[v]) for v in X}

        # 2. 标签处理
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        n_samples, n_classes = Y.shape
        view_names = list(X.keys())
        n_views = len(view_names)

        # 3. 初始化变量
        W = {v: np.zeros((X_std[v].shape[1], n_classes)) for v in view_names}
        Z_bar = np.zeros((n_samples, n_classes))
        U = np.zeros((n_samples, n_classes))

        B = {v: np.zeros_like(W[v]) for v in view_names}
        V = {v: np.zeros_like(W[v]) for v in view_names}

        # Cache Inverses
        Inv_matrices = {}
        for v in view_names:
            XtX = X_std[v].T @ X_std[v]
            d_v = XtX.shape[0]
            tmp = self.rho * XtX + self.mu_param * np.eye(d_v)
            Inv_matrices[v] = np.linalg.inv(tmp)

        # 4. ADMM Loop
        for k in range(self.max_iter):
            sum_XW = np.zeros_like(Z_bar)
            for v in view_names:
                sum_XW += X_std[v] @ W[v]
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
            for v in view_names:
                sum_XW_new += X_std[v] @ W[v]
            mean_XW_new = sum_XW_new / n_views
            Z_bar = (Y + self.rho * mean_XW_new + self.rho * U) / (n_views + self.rho)
            U = U + mean_XW_new - Z_bar

        # 【修改】传入 n_samples，使用 N/logN 逻辑
        return FitResult(selected_features=select_from_weights(W, X_std, n_samples=n_samples))


# ---------------------------------------------------------
# 2. HLRFS (Hypergraph Low-Rank)
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
        Sb = XtYc @ XtYc.T
        XLXt = X_concat.T @ (H_c @ L @ H_c) @ X_concat

        D_diag = np.eye(d)
        A = np.random.randn(d, self.r)

        for iter_ in range(self.max_iter):
            Sw = (Xc.T @ Xc) + (self.alpha * XLXt) + (self.lam * D_diag)
            Sw += 1e-6 * np.eye(d)

            vals, vecs = scipy.linalg.eigh(Sb, Sw)
            A = vecs[:, -self.r:]

            term1 = np.linalg.inv(A.T @ Sw @ A)
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

        # 【修改】传入 n_samples
        return FitResult(selected_features=select_from_weights(W_dict, X, n_samples=n))


# ---------------------------------------------------------
# 3. SCFS (Supervised Block Computing)
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

        X_indices = {}
        for idx, c in enumerate(classes):
            rows = np.where(y == c)[0]
            X_indices[idx] = rows

        W = {v: np.zeros((X_std[v].shape[1], n_classes)) for v in view_names}
        T_bar = {i: np.zeros((len(rows), n_classes)) for i, rows in X_indices.items()}
        P_bar = {i: np.zeros((len(rows), n_classes)) for i, rows in X_indices.items()}

        XtX_cache = {}
        for i in range(n_classes):
            XtX_cache[i] = {}
            rows = X_indices[i]
            for v in view_names:
                X_ij = X_std[v][rows]
                XtX_cache[i][v] = X_ij.T @ X_ij

        for k in range(self.max_iter):
            # Step 1: Update W
            for v in view_names:
                d_j = X_std[v].shape[1]
                w_row_norms = np.linalg.norm(W[v], axis=1)
                G_diag = 1.0 / (2.0 * w_row_norms + 1e-6)
                G_mat = np.diag(G_diag)

                Sum_XtX = np.zeros((d_j, d_j))
                for i in range(n_classes):
                    Sum_XtX += XtX_cache[i][v]

                for c_idx in range(n_classes):
                    w_col_norm = np.linalg.norm(W[v][:, c_idx])
                    q_val = 1.0 / (2.0 * w_col_norm + 1e-6)
                    Q_mat = q_val * np.eye(d_j)

                    LHS = 2 * self.l1 * G_mat + self.rho * Sum_XtX + 2 * self.l2 * Q_mat
                    RHS = np.zeros(d_j)

                    for i in range(n_classes):
                        rows = X_indices[i]
                        X_ij = X_std[v][rows]
                        pred_sum = np.zeros((len(rows), n_classes))
                        for v_temp in view_names:
                            pred_sum += X_std[v_temp][rows] @ W[v_temp]
                        pred_mean = pred_sum / n_views

                        T_col = T_bar[i][:, c_idx]
                        P_col = P_bar[i][:, c_idx]
                        XW_col = X_ij @ W[v][:, c_idx]
                        Mean_col = pred_mean[:, c_idx]

                        residual = T_col + XW_col - Mean_col - (P_col / self.rho)
                        RHS += self.rho * (X_ij.T @ residual)

                    W[v][:, c_idx] = np.linalg.solve(LHS + 1e-6 * np.eye(d_j), RHS)

            # Step 2: Update T
            for i in range(n_classes):
                rows = X_indices[i]
                Y_i = Y_full[rows]
                pred_sum = np.zeros_like(Y_i, dtype=float)
                for v in view_names:
                    pred_sum += X_std[v][rows] @ W[v]
                pred_mean = pred_sum / n_views

                H_i = pred_mean + (P_bar[i] / self.rho) - (Y_i / n_views)
                thresh = 1.0 / (2.0 * self.rho)
                h_norms = np.linalg.norm(H_i, axis=1, keepdims=True)
                h_norms[h_norms == 0] = 1e-10
                scale = np.maximum(0, 1 - thresh / h_norms)
                M_i = H_i * scale
                T_bar[i] = M_i + (Y_i / n_views)

            # Step 3: Update P
            for i in range(n_classes):
                rows = X_indices[i]
                pred_sum = np.zeros_like(T_bar[i])
                for v in view_names:
                    pred_sum += X_std[v][rows] @ W[v]
                pred_mean = pred_sum / n_views
                P_bar[i] = P_bar[i] + self.rho * (pred_mean - T_bar[i])

        # 【修改】传入 n_samples
        return FitResult(selected_features=select_from_weights(W, X_std, n_samples=n_samples))