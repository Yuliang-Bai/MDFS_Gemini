import numpy as np
from math import log, floor
from scipy.linalg import eigh, norm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Any

from ...base import BaseMethod, FitResult


# ==========================================
# 1. MCFL (Wang et al., 2013)
# ==========================================
class MCFL(BaseMethod):
    """
    Baseline: MCFL (Multi-View Clustering and Feature Learning)
    仅做最小改动：引入论文的 zeta 平滑（sqrt(||w||^2 + zeta)）替换原来的 (||w|| + eps)
    """

    def __init__(self, name="MCFL", n_clusters=3, gamma1=10.0, gamma2=10.0,
                 max_iter=50, tol=1e-4, zeta=1e-6, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_iter = max_iter
        self.tol = tol
        self.zeta = float(zeta) if zeta is not None else 1e-6
        if self.zeta <= 0:
            self.zeta = 1e-12  # 防御性：确保 >0

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None, verbose: bool = False) -> FitResult:
        sorted_keys = sorted(X.keys())
        X_list = []
        view_ranges = []
        start = 0
        n_samples = list(X.values())[0].shape[0]

        # 1) Data Prep
        for k in sorted_keys:
            data = np.nan_to_num(X[k])
            try:
                data_std = StandardScaler(with_mean=True, with_std=True).fit_transform(data)
            except Exception:
                data_std = data
            data_std = np.nan_to_num(data_std)
            X_list.append(data_std)
            dim = data_std.shape[1]
            view_ranges.append((start, start + dim))
            start += dim

        X_concat = np.hstack(X_list)          # (n, d)
        X_paper = X_concat.T                  # (d, n)
        d = X_concat.shape[1]
        c = self.n_clusters
        ones_n = np.ones((n_samples, 1))

        # Precompute
        XXt = X_paper @ X_paper.T             # (d, d)
        eye_d = np.eye(d) * 1e-6              # 数值稳定（保留你原来的做法）

        # 2) Init
        try:
            kmeans = KMeans(n_clusters=c, n_init=10, random_state=42)
            kmeans.fit(X_concat)
            F = np.eye(c)[kmeans.labels_]     # (n, c)
        except Exception:
            F = np.random.rand(n_samples, c)
            F = F / (np.sum(F, axis=1, keepdims=True) + 1e-8)

        A_ls = np.hstack([X_concat, ones_n])
        try:
            B_ls, _, _, _ = np.linalg.lstsq(A_ls, F, rcond=None)
        except Exception:
            B_ls = np.zeros((d + 1, c))
        B_ls = np.nan_to_num(B_ls)
        W = B_ls[:-1, :]                      # (d, c)
        b = B_ls[-1, :].reshape(c, 1)         # (c, 1)

        history = []
        if verbose:
            print(f"\n[MCFL Start] gamma1={self.gamma1}, gamma2={self.gamma2}, zeta={self.zeta:g}")

        # 3) Optimization
        for t in range(self.max_iter):
            W_prev = W.copy()

            # Update F via SVD (orthogonal Procrustes)
            M = np.nan_to_num(X_concat @ W + ones_n @ b.T)  # (n, c)
            try:
                U_svd, _, Vt_svd = np.linalg.svd(M, full_matrices=False)
                F = U_svd @ Vt_svd                           # (n, c)
            except Exception:
                pass

            # Update b (更一般闭式；数据中心化时等价于论文简化式)
            resid = F - (X_concat @ W)                        # (n, c)
            b = (resid.T @ ones_n) / n_samples                # (c, 1)

            # === ✅ 关键改进：reweight 使用 sqrt(||w||^2 + zeta) ===
            # \tilde D diag: 0.5 / sqrt(||w_row||^2 + zeta)
            row_norm2 = np.sum(W * W, axis=1)                 # (d,)
            d_tilde_diag = 0.5 / np.sqrt(row_norm2 + self.zeta)
            # 保留你原来的上限保护（可选，但不改结构）
            d_tilde_diag = np.minimum(d_tilde_diag, 1e5)

            W_new = np.zeros_like(W)
            for i in range(c):
                d_i_diag = np.zeros(d)
                w_col = W[:, i]

                # D_i block diag: each view block uses 0.5 / sqrt(||w_i^j||^2 + zeta)
                for (v_start, v_end) in view_ranges:
                    w_view = w_col[v_start:v_end]
                    view_norm2 = float(np.dot(w_view, w_view))
                    val = 0.5 / np.sqrt(view_norm2 + self.zeta)
                    val = min(val, 1e5)  # 保留原来的保护
                    d_i_diag[v_start:v_end] = val

                if verbose and t == 0 and i == 0:
                    print("[MCFL dbg] gamma1*d_i(max) =",
                          self.gamma1 * float(d_i_diag.max()),
                          "gamma2*d_tilde(max) =",
                          self.gamma2 * float(d_tilde_diag.max()))

                reg_vec = self.gamma1 * d_i_diag + self.gamma2 * d_tilde_diag
                reg_vec = np.nan_to_num(reg_vec, posinf=1e5)

                A = XXt + np.diag(reg_vec) + eye_d
                rhs = X_paper @ (F[:, i] - b[i, 0])
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    W_new[:, i] = sol
                except Exception:
                    W_new[:, i] = W[:, i]

            eta = 0.5  # 阻尼系数：0.3~0.7 都行，先用 0.5
            W = (1 - eta) * W_prev + eta * W_new
            W = np.nan_to_num(W)

            # Monitor
            diff = norm(W - W_prev) / (norm(W_prev) + 1e-8)
            row_norm = np.linalg.norm(W, axis=1)
            row_zero = float(np.mean(row_norm < 1e-8))  # 真正的“特征被干掉(按行)”比例

            w_max = float(np.max(np.abs(W))) if W.size else 0.0

            if verbose and (t % 10 == 0 or t == self.max_iter - 1):
                p50 = float(np.quantile(row_norm, 0.50))
                p90 = float(np.quantile(row_norm, 0.90))
                p99 = float(np.quantile(row_norm, 0.99))
                print(f"[MCFL] Iter {t:3d} | Diff: {diff:.6f} | "
                      f"RowZero: {row_zero:.2%} | "
                      f"Row(p50/p90/p99): {p50:.2e}/{p90:.2e}/{p99:.2e} | "
                      f"W_max: {w_max:.4f}")

            history.append({'iter': int(t), 'diff': float(diff),
                            'row_zero': row_zero, 'w_max': w_max})

            if diff < self.tol:
                if verbose:
                    print(f"[MCFL] Converged at iter {t}")
                break

        # 4) Feature Selection
        all_scores = np.nan_to_num(norm(W, axis=1))
        view_scores = {}
        for idx, k in enumerate(sorted_keys):
            v_start, v_end = view_ranges[idx]
            view_scores[k] = all_scores[v_start:v_end]

        selected = self._perform_screening_logic(X, view_scores, n_samples)
        return FitResult(selected_features=selected, model_state={"history": history, "W": W, "b": b, "F": F})


    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            scores = np.nan_to_num(all_scores_dict[name], nan=-np.inf)
            idx = np.argsort(-scores)[:k_m]
            selected[name] = idx.tolist()
        return selected


# ==========================================
# 2. MRAG (Jing et al., 2021)
# ==========================================
class MRAG(BaseMethod):
    """
    Baseline: MRAG (Robust Affinity Graph Representation Learning)
    """

    def __init__(self, name="MRAG", n_clusters=3, k_neighbors=10,
                 alpha=2.0, beta=0.05, gamma=1.0,
                 max_iter=50, use_hyperedge_weights=True, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.k = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.use_hyperedge_weights = use_hyperedge_weights

    @staticmethod
    def _orthonormalize(Y: np.ndarray) -> np.ndarray:
        Q, _ = np.linalg.qr(Y)
        return Q

    def _construct_hypergraph_laplacian(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        k_eff = min(self.k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X)
        distances, indices = nbrs.kneighbors(X)

        H = np.zeros((n, n))
        for i in range(n):
            H[indices[i], i] = 1.0

        W_diag = np.ones(n)
        if self.use_hyperedge_weights:
            for i in range(n):
                dists = distances[i, 1:]
                if dists.size == 0:
                    continue
                sigma = np.mean(dists) * (self.gamma if self.gamma > 0.01 else 1.0) + 1e-8
                sq_sum = np.sum(dists**2)
                w_i = np.exp(-sq_sum / (k_eff * sigma**2 + 1e-8))
                W_diag[i] = float(np.clip(w_i, 1e-8, 1.0))

        W_mat = np.diag(W_diag)
        Dv_diag = H @ W_diag
        De_diag = np.sum(H, axis=0)

        Dv_inv_sqrt = np.diag(1.0 / np.sqrt(Dv_diag + 1e-8))
        De_inv = np.diag(1.0 / (De_diag + 1e-8))

        Theta = Dv_inv_sqrt @ H @ W_mat @ De_inv @ H.T @ Dv_inv_sqrt
        L = np.eye(n) - Theta
        return (L + L.T) / 2

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None, verbose: bool = False) -> FitResult:
        sorted_keys = sorted(X.keys())
        n_samples = list(X.values())[0].shape[0]
        n_views = len(sorted_keys)
        c = self.n_clusters

        X_data = []
        L_sum = np.zeros((n_samples, n_samples))

        for k in sorted_keys:
            data = np.nan_to_num(X[k])
            try:
                data = StandardScaler().fit_transform(data)
            except Exception:
                pass
            data = np.nan_to_num(data)
            X_data.append(data)

            L_v = self._construct_hypergraph_laplacian(data)
            L_sum += np.nan_to_num(L_v)

        try:
            _, evecs = eigh(L_sum)
            start = 1 if n_samples > (c + 1) else 0
            Y = evecs[:, start:start + c]
        except Exception:
            Y = np.random.randn(n_samples, c)
        Y = self._orthonormalize(np.nan_to_num(Y))

        U_list = []
        for v in range(n_views):
            dim = X_data[v].shape[1]
            try:
                U_v, _, _, _ = np.linalg.lstsq(X_data[v], Y, rcond=None)
            except Exception:
                U_v = np.zeros((dim, c))
            U_list.append(np.nan_to_num(U_v))

        epsI_n = np.eye(n_samples) * 1e-6
        history = []

        if verbose:
            print(f"\n[MRAG Start] alpha={self.alpha}, beta={self.beta}")

        for t in range(self.max_iter):
            total_diff = 0.0
            M_sum = np.zeros((n_samples, c))
            u_sparsity_sum = 0.0

            for v in range(n_views):
                X_v = X_data[v]
                U_v = U_list[v]
                d_v = X_v.shape[1]

                XtX = X_v.T @ X_v
                u_norms = norm(U_v, axis=1) + 1e-8
                A_diag = np.minimum(0.5 / u_norms, 1e5)

                LHS = self.alpha * XtX + self.beta * np.diag(A_diag) + np.eye(d_v)*1e-6
                RHS = self.alpha * (X_v.T @ Y)

                U_prev = U_v.copy()
                try:
                    U_v, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
                except Exception:
                    pass
                U_v = np.nan_to_num(U_v)
                U_list[v] = U_v

                total_diff += float(norm(U_v - U_prev))
                M_sum += self.alpha * (X_v @ U_v)
                u_sparsity_sum += float(np.mean(np.abs(U_v) < 1e-4))

            LHS_Y = L_sum + (n_views * self.alpha) * np.eye(n_samples) + epsI_n
            try:
                Y_tilde, _, _, _ = np.linalg.lstsq(LHS_Y, M_sum, rcond=None)
            except Exception:
                Y_tilde = Y
            Y = self._orthonormalize(np.nan_to_num(Y_tilde))

            avg_sparsity = u_sparsity_sum / max(1, n_views)
            ortho_err = float(norm(Y.T @ Y - np.eye(c)))

            history.append({
                "iter": int(t),
                "diff": float(total_diff),
                "avg_u_sparsity": float(avg_sparsity),
                "ortho_err": ortho_err
            })

            if verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"[MRAG] Iter {t:3d} | Diff: {total_diff:.6f} | Avg U Sparsity: {avg_sparsity:.2%} | OrthoErr: {ortho_err:.2e}")

            if total_diff < 1e-4:
                if verbose:
                    print(f"[MRAG] Converged at iter {t}")
                break

        view_scores = {}
        for idx, k in enumerate(sorted_keys):
            scores = norm(U_list[idx], axis=1)
            view_scores[k] = np.nan_to_num(scores)

        selected = self._perform_screening_logic(X, view_scores, n_samples)
        return FitResult(selected_features=selected, model_state={"history": history})

    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            scores = np.nan_to_num(all_scores_dict[name], nan=-np.inf)
            idx = np.argsort(-scores)[:k_m]
            selected[name] = idx.tolist()
        return selected


# ==========================================
# 3. NSGL (Bai et al., 2020)
# ==========================================
class NSGL(BaseMethod):
    """
    Paper-replica: NSGL (Nonnegative Structured Graph Learning)
    - Objective: Eq.(11)
    - Alternate optimization: Algorithm 1
    - Updates: W(Eq.15), G(Eq.18), H(Eq.23), p(Eq.27)
    """

    def __init__(
        self,
        name="NSGL",
        n_clusters=3,
        k_neighbors=10,
        alpha=1.0,
        beta=1.0,
        lambda_=100.0,
        max_iter=30,
        view_weight_power=2.0,     # t in paper (default t=2)
        ortho_gamma=1e5,           # gamma in paper
        tau=1e-10,                 # small constant in reweighting for l2,1
        inner_irls_max=50,         # "until convergence" for Eq.(18)
        inner_irls_tol=1e-6,
        obj_tol=1e-6,
        random_state=0,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.c = int(n_clusters)
        self.k = int(k_neighbors)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lambda_)
        self.t_param = float(view_weight_power)
        self.gamma_param = float(ortho_gamma)
        self.tau = float(tau)

        self.max_iter = int(max_iter)
        self.inner_irls_max = int(inner_irls_max)
        self.inner_irls_tol = float(inner_irls_tol)
        self.obj_tol = float(obj_tol)
        self.random_state = int(random_state)

    # ----------------------------- basic utils -----------------------------

    @staticmethod
    def _pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
        ss = np.sum(A * A, axis=1)
        D = ss[:, None] + ss[None, :] - 2.0 * (A @ A.T)
        return np.maximum(D, 0.0)

    def _laplacian_from_W(self, W: np.ndarray) -> np.ndarray:
        # use symmetrized graph as in paper
        W_sym = 0.5 * (W + W.T)
        D = np.diag(np.sum(W_sym, axis=1))
        L = D - W_sym
        return 0.5 * (L + L.T)

    @staticmethod
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        """
        Euclidean projection onto simplex:
            min_w ||w - v||^2  s.t. w>=0, sum(w)=1
        """
        v = v.astype(float, copy=False)
        n = v.size
        if n == 1:
            return np.array([1.0], dtype=float)

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        if rho.size == 0:
            w = np.zeros_like(v)
            w[np.argmax(v)] = 1.0
            return w
        rho = int(rho[-1])
        theta = (cssv[rho] - 1.0) / float(rho + 1)
        w = np.maximum(v - theta, 0.0)

        s = w.sum()
        if s <= 0:
            w = np.zeros_like(v)
            w[np.argmax(v)] = 1.0
        else:
            w /= s
        return w

    # ----------------------------- paper updates -----------------------------

    def _update_w_eq15(self, D_total: np.ndarray) -> np.ndarray:
        """
        Eq.(15): for each i, w_i is the simplex projection of -d_i/(2*alpha) on kNN support.
        """
        n = D_total.shape[0]
        W = np.zeros((n, n), dtype=float)

        for i in range(n):
            d_i = D_total[i].copy()
            d_i[i] = np.inf

            kk = min(self.k, n - 1)
            idx = np.argpartition(d_i, kk)[:kk]
            idx = idx[np.argsort(d_i[idx])]
            d_sel = d_i[idx]

            v = -d_sel / (2.0 * self.alpha + 1e-12)
            w = self._proj_simplex(v)
            W[i, idx] = w

        return W

    def _init_H_from_W(self, W: np.ndarray, c: int) -> np.ndarray:
        """
        Algorithm 1: "calculate H based on W".
        Use spectral embedding (c smallest eigenvectors of Lw) + kmeans -> one-hot -> column normalize.
        This produces H>=0 and H^T H = I_c (after column normalization).
        """
        N = W.shape[0]
        Lw = self._laplacian_from_W(W)

        try:
            _, evecs = eigh(Lw, subset_by_index=[0, c - 1])
            U = evecs[:, :c]
        except Exception:
            U = np.random.RandomState(self.random_state).randn(N, c)

        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

        try:
            km = KMeans(n_clusters=c, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(U)
        except Exception:
            rng = np.random.RandomState(self.random_state)
            labels = rng.randint(0, c, size=N)

        H = np.zeros((N, c), dtype=float)
        H[np.arange(N), labels] = 1.0

        # column normalize => H^T H = I
        H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-12)
        return H

    def _update_G_eq18(self, X_concat: np.ndarray, H: np.ndarray, G_init: np.ndarray):
        """
        Eq.(18) with iterative reweighting for l2,1:
            G = (X X^T + Gamma)^{-1} X H
        where Gamma is diagonal with Gamma_ii = 1/(2*||g_i|| + tau).
        Run until convergence (paper: "until convergence").
        """
        d_total = X_concat.shape[0]
        XXT = X_concat @ X_concat.T  # (d,d)
        epsI = np.eye(d_total) * 1e-10

        G = G_init.copy()
        Gamma_diag = np.ones(d_total, dtype=float)

        for _ in range(self.inner_irls_max):
            G_prev = G

            g_norms = norm(G, axis=1)
            Gamma_diag = 1.0 / (2.0 * (g_norms + self.tau))

            LHS = XXT + np.diag(Gamma_diag) + epsI
            RHS = X_concat @ H  # (d,c)

            try:
                G = np.linalg.solve(LHS, RHS)
            except Exception:
                G, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)

            G = np.nan_to_num(G)

            rel = norm(G - G_prev) / (norm(G_prev) + 1e-12)
            if rel < self.inner_irls_tol:
                break

        return G, Gamma_diag

    def _update_H_eq23(self, W: np.ndarray, X_concat: np.ndarray, H: np.ndarray, Gamma_diag: np.ndarray) -> np.ndarray:
        """
        Eq.(23) multiplicative update for H after forming:
            M = 2*lambda*Lw + beta*(I - X^T (X X^T + Gamma)^{-1} X)
        then:
            H <- H ⊙ (gamma*H) / (M H + gamma H H^T H)
        followed by column normalization.
        """
        N = H.shape[0]
        d_total = X_concat.shape[0]
        I_N = np.eye(N)

        Lw = self._laplacian_from_W(W)

        XXT = X_concat @ X_concat.T
        A = XXT + np.diag(Gamma_diag) + np.eye(d_total) * 1e-10

        # compute S = X^T A^{-1} X
        try:
            AinvX = np.linalg.solve(A, X_concat)  # (d,N)
        except Exception:
            AinvX, _, _, _ = np.linalg.lstsq(A, X_concat, rcond=None)

        S = X_concat.T @ AinvX
        S = 0.5 * (S + S.T)  # symmetry for numerical stability

        M = 2.0 * self.lam * Lw + self.beta * (I_N - S)
        M = 0.5 * (M + M.T)

        numer = self.gamma_param * H
        denom = (M @ H) + self.gamma_param * (H @ (H.T @ H))

        # exact formula; clamp only to avoid division-by-zero
        denom = np.maximum(denom, 1e-12)

        H_new = H * (numer / denom)
        H_new = np.maximum(H_new, 0.0)

        # normalize columns so (H^T H)_ii = 1
        H_new = H_new / (np.linalg.norm(H_new, axis=0, keepdims=True) + 1e-12)
        return H_new

    def _update_p_eq27(self, Dist_X: list, W: np.ndarray) -> np.ndarray:
        """
        Eq.(27): p_v ∝ (1/o_v)^{1/(t-1)},  o_v = sum_{i,j} ||x_i^v - x_j^v||^2 w_ij
        """
        n_views = len(Dist_X)
        objs = np.array([np.sum(Dist_X[v] * W) for v in range(n_views)], dtype=float)
        objs = np.maximum(objs, 1e-12)

        power_factor = 1.0 / (self.t_param - 1.0 + 1e-12)
        terms = np.power(1.0 / objs, power_factor)
        p = terms / (np.sum(terms) + 1e-12)
        return np.nan_to_num(p)

    def _objective_eq11(self, Dist_X: list, W: np.ndarray, H: np.ndarray, X_concat: np.ndarray, G: np.ndarray, p: np.ndarray) -> float:
        """
        Eq.(11) objective value (for convergence check).
        """
        term_graph = 0.0
        for v in range(len(Dist_X)):
            term_graph += p[v] * (np.sum(Dist_X[v] * W) + self.alpha * np.sum(W * W))

        Lw = self._laplacian_from_W(W)
        term_struct = 2.0 * self.lam * np.trace(H.T @ Lw @ H)

        # ||X^T G - H||_F^2 + ||G||_{2,1}
        XTG = X_concat.T @ G
        term_reg = self.beta * (np.sum((XTG - H) ** 2) + np.sum(norm(G, axis=1)))

        return float(term_graph + term_struct + term_reg)

    # ----------------------------- main fit -----------------------------

    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None, verbose: bool = False) -> FitResult:
        sorted_keys = sorted(X.keys())
        n_views = len(sorted_keys)
        N = list(X.values())[0].shape[0]
        c = self.c

        # (1) prepare per-view data matrices and concat X (d x N)
        Xv_list = []
        d_list = []
        for k in sorted_keys:
            data = np.nan_to_num(X[k]).astype(float)
            # NOTE: paper does not mandate a specific scaling; keep raw by default.
            Xv = data.T  # (d_v, N)
            Xv_list.append(Xv)
            d_list.append(Xv.shape[0])

        d_total = sum(d_list)
        X_concat = np.vstack(Xv_list)  # (d_total, N)

        # (2) precompute per-view pairwise distances in sample space (N x N)
        Dist_X = []
        for v in range(n_views):
            Data = Xv_list[v].T  # (N, d_v)
            D = self._pairwise_sq_dists(Data)
            Dist_X.append(np.nan_to_num(D))

        # (3) init p, G, W, H (Algorithm 1)
        p = np.ones(n_views, dtype=float) / float(n_views)
        G = np.zeros((d_total, c), dtype=float)

        # init W using p-weighted distances (H not used at init)
        Dist_X_weighted = np.zeros((N, N), dtype=float)
        for v in range(n_views):
            Dist_X_weighted += p[v] * Dist_X[v]
        W = self._update_w_eq15(Dist_X_weighted)

        # init H based on W
        H = self._init_H_from_W(W, c)

        history = []
        prev_obj = None

        if verbose:
            print(f"\n[NSGL Start] alpha={self.alpha}, beta={self.beta}, lambda={self.lam}, t={self.t_param}, gamma={self.gamma_param}")

        for it in range(self.max_iter):
            # ---- (1) update W : Eq.(15)
            Dist_H = self._pairwise_sq_dists(H)
            D_total = np.zeros((N, N), dtype=float)
            for v in range(n_views):
                D_total += p[v] * Dist_X[v]
            D_total += self.lam * Dist_H
            D_total = np.nan_to_num(D_total)
            W = self._update_w_eq15(D_total)

            # ---- (2) update G : Eq.(18) (until convergence inside)
            G_prev = G.copy()
            G, Gamma_diag = self._update_G_eq18(X_concat, H, G)

            # ---- (3) update H : Eq.(23)
            H = self._update_H_eq23(W, X_concat, H, Gamma_diag)

            # ---- (4) update p : Eq.(27)
            p = self._update_p_eq27(Dist_X, W)

            # ---- monitor / stop
            obj = self._objective_eq11(Dist_X, W, H, X_concat, G, p)
            g_diff = norm(G - G_prev) / (norm(G_prev) + 1e-12)
            w_density = float(np.mean(W > 1e-12))

            history.append({
                "iter": int(it),
                "obj": float(obj),
                "g_diff": float(g_diff),
                "w_density": w_density,
                "p": p.copy(),
            })

            if verbose and (it % 5 == 0 or it == self.max_iter - 1):
                print(f"[NSGL] Iter {it:3d} | obj: {obj:.6e} | g_diff: {g_diff:.3e} | w_density: {w_density:.4f} | p: {np.round(p, 4)}")

            if prev_obj is not None:
                rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
                if rel < self.obj_tol:
                    if verbose:
                        print(f"[NSGL] Converged at iter {it} (rel_obj={rel:.3e})")
                    break
            prev_obj = obj

        # feature scores: ||g_i||_2
        all_scores = norm(G, axis=1)
        all_scores = np.nan_to_num(all_scores)

        view_scores = {}
        curr = 0
        for idx, k in enumerate(sorted_keys):
            dim = d_list[idx]
            view_scores[k] = all_scores[curr: curr + dim]
            curr += dim

        selected = self._perform_screening_logic(X, view_scores, N)

        return FitResult(
            selected_features=selected,
            model_state={
                "history": history,
                "W": W,
                "H": H,
                "G": G,
                "p": p,
            }
        )

    def _perform_screening_logic(self, X_dict, all_scores_dict, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            scores = np.nan_to_num(all_scores_dict[name], nan=-np.inf)
            idx = np.argsort(-scores)[:k_m]
            selected[name] = idx.tolist()
        return selected
