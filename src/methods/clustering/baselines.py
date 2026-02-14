import numpy as np
from scipy.linalg import eigh, norm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Any

# 假设你的 base 文件名为 base.py
from ...base import BaseMethod, FitResult

# ==========================================
# 1. MCFL (Wang et al., 2013)  [Interface kept]
# ==========================================
class MCFL(BaseMethod):
    """
    Baseline: MCFL (Multi-View Clustering and Feature Learning)
    Reference: Wang et al., "Multi-view clustering and feature learning via structured sparsity", 2013.
    """

    def __init__(self, name="MCFL", n_clusters=3, gamma1=10.0, gamma2=10.0,
                 max_iter=50, tol=1e-4, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None) -> FitResult:
        # 1) Data prep
        sorted_keys = sorted(X.keys())
        X_list = []
        view_ranges = []
        start = 0
        n_samples = list(X.values())[0].shape[0]

        for k in sorted_keys:
            data_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X[k])
            X_list.append(data_std)
            dim = data_std.shape[1]
            view_ranges.append((start, start + dim))
            start += dim

        X_concat = np.hstack(X_list)          # (n, d)
        X_paper = X_concat.T                  # (d, n)
        d = X_concat.shape[1]
        c = self.n_clusters
        ones_n = np.ones((n_samples, 1))

        # Precompute XX^T (d x d)
        XXt = X_paper @ X_paper.T
        eye_d = np.eye(d) * 1e-8

        # 2) Initialize F via kmeans, then initialize (W,b) by least squares:
        #    min || X^T W + 1 b^T - F ||_F^2  (paper Algorithm 1 Step 1)
        kmeans = KMeans(n_clusters=c, n_init=10, random_state=42)
        kmeans.fit(X_concat)
        F = np.eye(c)[kmeans.labels_]         # (n, c)

        A_ls = np.hstack([X_concat, ones_n])  # (n, d+1)
        B_ls, *_ = np.linalg.lstsq(A_ls, F, rcond=None)  # (d+1, c)
        W = B_ls[:-1, :]                      # (d, c)
        b = B_ls[-1, :].reshape(c, 1)         # (c, 1)

        # 3) Iterative optimization
        for t in range(self.max_iter):
            W_prev = W.copy()

            # Update F (Theorem 1 style: orthogonal Procrustes on M = X^T W + 1 b^T)
            M = X_concat @ W + ones_n @ b.T  # (n, c)
            U_svd, _, Vt_svd = np.linalg.svd(M, full_matrices=False)
            F = U_svd @ Vt_svd              # (n, c), orthonormal columns

            # Update b from closed form of min || X^T W + 1 b^T - F ||_F^2:
            # b^T = (1^T (F - X^T W))/n
            resid = F - (X_concat @ W)       # (n, c)
            b = (resid.T @ ones_n) / n_samples  # (c, 1)

            # Update W (Eq 6): column-wise solve
            w_row_norms = norm(W, axis=1) + 1e-8
            d_tilde_diag = 0.5 / w_row_norms  # (d,)

            W_new = np.zeros_like(W)
            for i in range(c):
                # D^i: view-wise block constant (per view norm)
                d_i_diag = np.zeros(d)
                w_col = W[:, i]
                for (v_start, v_end) in view_ranges:
                    w_view = w_col[v_start:v_end]
                    d_i_diag[v_start:v_end] = 0.5 / (norm(w_view) + 1e-8)

                reg_vec = self.gamma1 * d_i_diag + self.gamma2 * d_tilde_diag
                A = XXt + np.diag(reg_vec) + eye_d
                rhs = X_paper @ (F[:, i] - b[i, 0])  # (d,)
                try:
                    W_new[:, i] = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    W_new[:, i] = np.linalg.lstsq(A, rhs, rcond=None)[0]

            W = W_new
            if norm(W - W_prev) / (norm(W_prev) + 1e-8) < self.tol:
                break

        # 4) Feature selection (keep your logic)
        all_scores = norm(W, axis=1)  # row norms => feature importance

        view_scores = {}
        for idx, k in enumerate(sorted_keys):
            v_start, v_end = view_ranges[idx]
            view_scores[k] = all_scores[v_start:v_end]

        selected = self._perform_screening_logic(X, view_scores, n_samples)
        return FitResult(selected_features=selected)

    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            idx = np.argsort(-all_scores_dict[name])[:k_m]
            selected[name] = idx.tolist()
        return selected


# ==========================================
# 2. MRAG (Jing et al., 2021)  [Interface kept]
#    Fix: alternate update for (U,Y) in RHLC, enforce Y^T Y = I
# ==========================================
class MRAG(BaseMethod):
    """
    Baseline: MRAG (Robust Affinity Graph Representation Learning)
    Implementation: RHLC (Eq 7) logic for feature scores.
    """

    def __init__(self, name="MRAG", n_clusters=3, k_neighbors=10,
                 alpha=2.0, beta=0.05, gamma=0.001,
                 max_iter=50, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.k = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter

    @staticmethod
    def _orthonormalize(Y: np.ndarray) -> np.ndarray:
        # QR -> orthonormal columns
        Q, _ = np.linalg.qr(Y)
        return Q

    def _construct_hypergraph_laplacian(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        _, indices = nbrs.kneighbors(X)

        # incidence H: each sample i forms a hyperedge e_i containing its (k+1) neighbors (incl. itself)
        H = np.zeros((n, n))
        for i in range(n):
            H[indices[i], i] = 1.0

        Dv = np.diag(np.sum(H, axis=1))
        De = np.diag(np.sum(H, axis=0))

        Dv_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Dv) + 1e-8))
        De_inv = np.diag(1.0 / (np.diag(De) + 1e-8))

        # Paper has Theta = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}; here W=I for simplicity
        Theta = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
        L = np.eye(n) - Theta
        return (L + L.T) / 2

    def _solve_rhlc(self, X: np.ndarray) -> np.ndarray:
        """
        RHLC alternating optimization (Eq 7):
            min_{U,Y} Tr(Y^T L Y) + alpha||XU - Y||_F^2 + beta||U||_{2,1}
            s.t. Y^T Y = I
        We implement:
          - U update: (alpha X^T X + beta A)U = alpha X^T Y, A_ii = 1/(2||u_i||)
          - Y update: solve (L + alpha I)Y = alpha XU, then re-orthonormalize columns
        """
        n, d = X.shape
        c = self.n_clusters
        L = self._construct_hypergraph_laplacian(X)

        # init Y from smallest non-trivial eigenvectors of L, then orthonormalize
        evals, evecs = eigh(L)
        start = 1 if n > (c + 1) else 0
        Y = evecs[:, start:start + c]
        Y = self._orthonormalize(Y)

        # init U by ridge-like solve (avoid U=0 leading to A blow-up)
        U = np.linalg.lstsq(X, Y, rcond=None)[0]  # (d, c)

        XtX = X.T @ X
        epsI_d = np.eye(d) * 1e-8
        epsI_n = np.eye(n) * 1e-8

        for t in range(self.max_iter):
            U_prev = U.copy()

            # Update A from row norms of U (feature sparsity)
            u_norms = norm(U, axis=1) + 1e-8
            A_diag = 0.5 / u_norms  # (d,)

            # Update U
            LHS = self.alpha * XtX + self.beta * np.diag(A_diag) + epsI_d
            RHS = self.alpha * (X.T @ Y)
            try:
                U = np.linalg.solve(LHS, RHS)
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

            # Update Y (solve then enforce Y^T Y = I)
            B = self.alpha * (X @ U)  # (n, c)
            try:
                Y_tilde = np.linalg.solve(L + self.alpha * np.eye(n) + epsI_n, B)
            except np.linalg.LinAlgError:
                Y_tilde = np.linalg.lstsq(L + self.alpha * np.eye(n) + epsI_n, B, rcond=None)[0]
            Y = self._orthonormalize(Y_tilde)

            if norm(U - U_prev) / (norm(U_prev) + 1e-8) < 1e-4:
                break

        # Feature scores: row norms of U (each row corresponds to a feature)
        return norm(U, axis=1)

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None) -> FitResult:
        sorted_keys = sorted(X.keys())
        n_samples = list(X.values())[0].shape[0]

        view_scores = {}
        for k in sorted_keys:
            data = StandardScaler().fit_transform(X[k])
            view_scores[k] = self._solve_rhlc(data)

        selected = self._perform_screening_logic(X, view_scores, n_samples)
        return FitResult(selected_features=selected)

    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            idx = np.argsort(-all_scores_dict[name])[:k_m]
            selected[name] = idx.tolist()
        return selected


# ==========================================
# 3. NSGL (Bai et al., 2020)  [Interface kept]
#    Fix: L_W definition, and updates of W/G/H/p by Eq (15)(18)(23)(27)
# ==========================================
class NSGL(BaseMethod):
    """
    Baseline: NSGL (Nonnegative Structured Graph Learning)
    Reference: Bai et al., Neurocomputing 2020.
    """

    def __init__(self, name="NSGL", n_clusters=3, k_neighbors=5,
                 alpha=1.0, beta=1.0, lambda_=0.1,
                 max_iter=50, **kwargs):
        super().__init__(name, **kwargs)
        self.c = n_clusters
        self.k = k_neighbors
        self.alpha = alpha      # corresponds to alpha in Eq (1)/(15)
        self.beta = beta        # corresponds to beta in Eq (19)/(21)
        self.lam = lambda_      # corresponds to lambda in Eq (11)/(21)
        self.max_iter = max_iter

    @staticmethod
    def _pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
        # A: (N, d)
        ss = np.sum(A * A, axis=1)
        D = ss[:, None] + ss[None, :] - 2.0 * (A @ A.T)
        return np.maximum(D, 0.0)

    def _laplacian_from_W(self, W: np.ndarray) -> np.ndarray:
        # Paper: L_W = D - (W^T + W)/2,  D_ii = sum_j (w_ij + w_ji)/2
        W_sym = 0.5 * (W + W.T)
        D = np.diag(np.sum(W_sym, axis=1))
        L = D - W_sym
        return (L + L.T) / 2

    def _update_w_qp(self, D_x_weighted: np.ndarray, D_h: np.ndarray,
                     alpha: float, lambda_val: float) -> np.ndarray:
        """
        Eq (15): for each i solve
          min || w_i + (1/(2alpha)) d_i ||^2
          s.t. w_i^T 1 = 1, w_i >= 0
        with d_ij = (sum_v p_v dx^v_ij) + lambda * dh_ij  (Eq (14))
        We implement k-NN truncation (as in Algorithm 1 input k), but enforce sum=1.
        """
        n = D_x_weighted.shape[0]
        W = np.zeros((n, n), dtype=float)
        D_total = D_x_weighted + lambda_val * D_h

        for i in range(n):
            d_i = D_total[i].copy()
            d_i[i] = np.inf  # exclude self

            # pick k smallest distances
            idx = np.argsort(d_i)[:self.k]
            d_sel = d_i[idx]

            # eta formula (common closed form used with fixed neighbor set)
            eta = (1.0 + np.sum(d_sel) / (2.0 * alpha)) / float(len(idx))
            w = eta - d_sel / (2.0 * alpha)
            w = np.maximum(w, 0.0)

            s = w.sum()
            if s <= 1e-12:
                # fallback: uniform on chosen neighbors
                w = np.ones_like(w) / float(len(w))
            else:
                w = w / s

            W[i, idx] = w

        return W

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None) -> FitResult:
        sorted_keys = sorted(X.keys())
        n_views = len(sorted_keys)

        N = list(X.values())[0].shape[0]
        c = self.c

        # ---- build per-view matrices Xv in paper shape (d_v, N)
        # Use MinMaxScaler to keep nonnegative scale consistent with H>=0/W>=0 setting.
        Xv_list = []
        d_list = []
        for k in sorted_keys:
            X_scaled = MinMaxScaler().fit_transform(X[k])  # (N, d_v)
            Xv = X_scaled.T                                # (d_v, N)
            Xv_list.append(Xv)
            d_list.append(Xv.shape[0])

        d_total = sum(d_list)
        X_concat = np.vstack(Xv_list)  # (d_total, N)

        # ---- precompute Dist_X^v (N x N)
        Dist_X = []
        for v in range(n_views):
            Data = Xv_list[v].T  # (N, d_v)
            Dist_X.append(self._pairwise_sq_dists(Data))

        # ---- init p, G, W, H (Algorithm 1 Step 1)
        p = np.ones(n_views) / n_views
        G = np.zeros((d_total, c), dtype=float)

        Dist_X_weighted = np.zeros((N, N), dtype=float)
        for v in range(n_views):
            Dist_X_weighted += p[v] * Dist_X[v]

        W = self._update_w_qp(Dist_X_weighted, np.zeros((N, N)), self.alpha, 0.0)
        Lw = self._laplacian_from_W(W)
        _, evecs = eigh(Lw, subset_by_index=[0, c - 1])
        H = np.abs(evecs)  # nonnegative init
        H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-8)

        # ---- optimization (Eq 11) alternate updates
        epsI_d = np.eye(d_total) * 1e-6
        gamma_param = 1e5  # Lagrange multiplier gamma in Eq (20), should be large

        for t in range(self.max_iter):
            # (1) Update W by Eq (15) with di = sum_v p_v dx^v + lambda * dh
            # dh_ij = ||h_i - h_j||^2
            Dist_H = self._pairwise_sq_dists(H)  # (N, N)
            Dist_X_weighted = np.zeros((N, N), dtype=float)
            for v in range(n_views):
                Dist_X_weighted += p[v] * Dist_X[v]
            W = self._update_w_qp(Dist_X_weighted, Dist_H, self.alpha, self.lam)

            # (2) Update G by Eq (18) with Phi from current G (Eq (17))
            g_norms = norm(G, axis=1) + 1e-8
            Phi = np.diag(0.5 / g_norms)  # (d_total, d_total)

            XXT = X_concat @ X_concat.T
            LHS = XXT + Phi + epsI_d
            RHS = X_concat @ H  # (d_total, c)
            try:
                G = np.linalg.solve(LHS, RHS)
            except np.linalg.LinAlgError:
                G = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

            # (3) Update H by Eq (23)
            # M = 2*lambda*Lw + beta*(I - X^T (XX^T + Phi)^(-1) X)  (Eq 21)
            Lw = self._laplacian_from_W(W)
            try:
                P = np.linalg.solve(LHS, X_concat)  # (d_total, N)
            except np.linalg.LinAlgError:
                P = np.linalg.lstsq(LHS, X_concat, rcond=None)[0]
            S = X_concat.T @ P  # (N, N) = X^T (XX^T + Phi)^(-1) X

            M = 2.0 * self.lam * Lw + self.beta * (np.eye(N) - S)

            denom = (M @ H) + gamma_param * (H @ (H.T @ H))
            denom = np.maximum(denom, 1e-12)
            numer = gamma_param * H
            H = H * (numer / denom)

            # enforce H>=0 and normalize columns to satisfy approx H^T H = I
            H = np.maximum(H, 1e-12)
            H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-8)

            # (4) Update p by Eq (27) (take t=2 -> p_v ∝ 1/o_v)
            o_v = np.array([np.sum(Dist_X[v] * W) for v in range(n_views)], dtype=float)
            o_v = np.maximum(o_v, 1e-12)
            inv = 1.0 / o_v
            p = inv / np.sum(inv)

        # ---- feature ranking from row norms of G (Algorithm 1 Step 8-9)
        all_scores = norm(G, axis=1)  # (d_total,)

        view_scores = {}
        curr = 0
        for idx, k in enumerate(sorted_keys):
            dim = d_list[idx]
            view_scores[k] = all_scores[curr: curr + dim]
            curr += dim

        selected = self._perform_screening_logic(X, view_scores, N)
        return FitResult(selected_features=selected)

    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            idx = np.argsort(-all_scores_dict[name])[:k_m]
            selected[name] = idx.tolist()
        return selected
