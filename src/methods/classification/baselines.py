import numpy as np
from typing import Dict, List, Optional
from math import floor, log
from scipy.sparse import csgraph
import scipy.linalg
from sklearn.neighbors import kneighbors_graph
from ...base import BaseMethod, FitResult


# ==========================================
# 辅助函数 (Utility Function)
# ==========================================
def select_from_weights(weights_dict: Dict[str, np.ndarray], X_dict: Dict[str, np.ndarray], n_samples: int) -> Dict[
    str, List[int]]:
    """
    根据权重字典进行特征排序和筛选
    weights_dict: {view_name: weight_matrix}
    """
    k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
    total_dims = sum([v.shape[1] for v in X_dict.values()])

    selected = {}
    for name, W in weights_dict.items():
        # 计算特征重要性分数: L2 Norm (对应 L2,1 范数的行范数)
        if W.ndim > 1:
            scores = np.linalg.norm(W, axis=1)
        else:
            scores = np.abs(W)

        dim = X_dict[name].shape[1]
        # 按特征维度比例分配筛选数量
        k_m = max(1, int(k_total * (dim / total_dims)))

        # 降序排列取 Top-k
        idx = np.argsort(scores)[::-1][:k_m]
        selected[name] = idx.tolist()

    return selected


def soft_threshold_row(B: np.ndarray, thresh: float) -> np.ndarray:
    """
    行级软阈值算子 (Row-wise Soft Thresholding for L2,1 Norm)
    对应论文公式 (30)
    """
    # 计算每一行的 L2 范数
    row_norms = np.linalg.norm(B, axis=1, keepdims=True)
    # 避免除以零
    scale = np.maximum(0, 1 - thresh / (row_norms + 1e-10))
    return B * scale


# ==========================================
# Baseline 1: SMVFS (ADMM)
# ==========================================
class SMVFS(BaseMethod):
    """
    A Sharing Multi-view Feature Selection via ADMM (Lin et al., 2019)
    [cite_start]Implementation strictly follows Algorithm 1 in the paper[cite: 842].
    Objective: min_W 0.5 * || sum(XvWv) - Y ||_F^2 + lambda * sum(||Wv||_2,1)
    """

    def __init__(self, name="SMVFS", alpha=0.1, rho=1.0, mu=1.0, max_iter=20, inner_iter=10, **kwargs):
        super().__init__(name, **kwargs)
        self.lambda_ = alpha  # 对应论文中的 lambda (正则化参数)
        self.rho = rho  # 对应论文中的 rho (外层惩罚参数)
        self.mu = mu  # 对应论文中的 mu_v (内层惩罚参数，简化设为统一值)
        self.max_iter = max_iter
        self.inner_iter = inner_iter  # 对应 Algorithm 1 Step 11 'Until suber < 60' (用固定次数近似)

        self.cached_inv = {}  # 缓存矩阵逆

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # --- 1. 初始化与预处理 ---
        view_names = sorted(X.keys())
        n_samples = len(y)

        # 标签处理 Y (n x c)
        # [cite_start]修正: 严格遵循论文 Eq(1) 和 Section 2.1 对 Y 的定义 [cite: 95, 96]
        # "If xj belongs to the pth class, then yj^p is 1; otherwise, yj^p is 0."
        n_classes = len(np.unique(y))

        # 即使是二分类，也构建 One-Hot 矩阵以匹配论文符号定义 (W in R^{d x c})
        if y.ndim == 1:
            classes = np.unique(y)
            n_classes_actual = len(classes)
            # 映射 label 到 0..k-1
            label_map = {label: i for i, label in enumerate(classes)}
            y_mapped = np.array([label_map[label] for label in y])
            Y = np.eye(n_classes_actual)[y_mapped]
        else:
            Y = y  # 假设已经是 One-Hot

        c_dim = Y.shape[1]

        # 变量初始化
        # W_dict: {view: (d_v, c)}
        W = {v: np.zeros((X[v].shape[1], c_dim)) for v in view_names}

        # 全局共享变量
        Z_bar = np.zeros((n_samples, c_dim))  # mean(Z_v)
        U = np.zeros((n_samples, c_dim))  # 对偶变量 U
        XW_bar = np.zeros((n_samples, c_dim))  # mean(X_v W_v)

        # 内层循环辅助变量 (针对每个view)
        # 论文 Step 3 提到 B^0 是 random matrix，这里使用全0进行 Warm start，工程上更稳定
        B = {v: np.zeros_like(W[v]) for v in view_names}
        V = {v: np.zeros_like(W[v]) for v in view_names}

        # [cite_start]--- 2. 预计算矩阵逆 (避免在大循环中重复计算) [cite: 230] ---
        # Cache (rho * XtX + mu * I)^(-1)
        # 对应 Eq (28) 中的逆矩阵部分
        for v in view_names:
            X_v = X[v]
            d_v = X_v.shape[1]
            XtX = X_v.T @ X_v
            # 公式 (28): (rho * XtX + mu * I)^(-1)
            matrix_to_inv = self.rho * XtX + self.mu * np.eye(d_v)
            self.cached_inv[v] = np.linalg.inv(matrix_to_inv)

        # --- 3. 外层循环 (ADMM Main Loop) ---
        for k in range(self.max_iter):

            # --- 3.1 准备阶段 ---
            # 计算当前的 XW_bar (用于 Step 4 计算 A_v)
            XW_sum = np.zeros_like(Z_bar)
            for v in view_names:
                XW_sum += X[v] @ W[v]
            XW_bar_prev = XW_sum / len(view_names)

            # --- 3.2 更新每个 View 的 W_v (并行子问题) ---
            # 对应论文公式 (19) 和 Algorithm 1 Step 4-11

            for v in view_names:
                X_v = X[v]
                # [cite_start]Step 4: Calculate A_v^k (Note: Paper uses symbol roughly looking like 'A') [cite: 191]
                # A_v^k = Z_bar^k + X_v W_v^k - XW_bar^k - U^k
                # 注意公式 (191) 中定义 A_v^k
                A_v = Z_bar + X_v @ W[v] - XW_bar_prev - U

                # 内层循环: 求解子问题 min_Wv (公式 22)
                for s in range(self.inner_iter):
                    # [cite_start]Step 6: Update W_v [cite: 208]
                    # W_v = (rho XtX + mu I)^-1 * (rho Xt A_v + mu (B - V))
                    term1 = self.rho * (X_v.T @ A_v)
                    term2 = self.mu * (B[v] - V[v])
                    W[v] = self.cached_inv[v] @ (term1 + term2)

                    # [cite_start]Step 7: Update B (Soft Thresholding) [cite: 242]
                    # B = argmin lambda ||B||_2,1 + mu/2 ||W - B + V||_F^2
                    # 这是一个 proximal operator 问题，对应公式 (30)
                    T = W[v] + V[v]
                    threshold = self.lambda_ / self.mu
                    B[v] = soft_threshold_row(T, threshold)

                    # [cite_start]Step 8: Update V [cite: 206]
                    # V^{s+1} = V^s + W^{s+1} - B^{s+1}
                    V[v] = V[v] + W[v] - B[v]

            # --- 3.3 更新全局变量 ---

            # 重新计算 XW_bar (用新的 W)
            XW_sum = np.zeros_like(Z_bar)
            for v in view_names:
                XW_sum += X[v] @ W[v]
            XW_bar = XW_sum / len(view_names)

            # [cite_start]Step 14: Update Z_bar [cite: 218, 842]
            # Z_bar = (1 / (N + rho)) * (Y + rho * XW_bar + rho * U)
            # 对应公式 (31)
            N = len(view_names)
            Z_bar = (Y + self.rho * XW_bar + self.rho * U) / (N + self.rho)

            # [cite_start]Step 15: Update U [cite: 177, 842]
            # U^{k+1} = -Z_bar^{k+1} + XW_bar^{k+1} + U^k
            # 对应公式 (21)
            U = U + XW_bar - Z_bar

            # (可选) Step 16: 计算收敛误差 er 用于提前停止
            # 这里简化为固定迭代次数

        # --- 4. 特征筛选 ---
        # 使用最终的 W 进行特征筛选
        selected = select_from_weights(W, X, n_samples)

        return FitResult(selected_features=selected)


# ==========================================
# Baseline 2: HLRFS (Hypergraph / Graph Reg)
# ==========================================
class HLRFS(BaseMethod):
    """
    Hypergraph Low-Rank Feature Selection (Cheng et al., 2017)
    Implementation follows the low-rank decomposition W=AB and hypergraph regularization.
    Objective: min ||Y - XAB - eb||_F^2 + alpha * tr(B'A'XLX'AB) + lambda * ||AB||_2,1
    """

    def __init__(self, name="HLRFS", alpha=1.0, lambda_=0.1, r=5, n_neighbors=5, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.alpha = alpha  # Graph regularization parameter (对应论文中的 alpha)
        self.lambda_ = lambda_  # Sparsity parameter (对应论文中的 lambda)
        self.r = r  # Rank of the decomposition (对应论文中的 r)
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter

    def _construct_hypergraph_laplacian(self, X: np.ndarray) -> np.ndarray:
        """
        Construct Hypergraph Laplacian L [cite: 171]
        Method: kNN-based hyperedge construction. Each sample i generates a hyperedge e_i
        containing i and its k-nearest neighbors.
        """
        n_samples = X.shape[0]
        # 1. Construct Incidence Matrix H (n_nodes x n_edges)
        # In this construction, n_edges = n_samples
        H_inc = np.zeros((n_samples, n_samples))

        # 使用 sklearn 计算 kNN
        # [cite_start]include_self=True 因为超边通常包含中心节点自己 [cite: 164-169]
        graph = kneighbors_graph(X, self.n_neighbors, mode='connectivity', include_self=True)
        H_inc = graph.toarray().T  # H_inc[i, j] = 1 if node i is in hyperedge j

        # [cite_start]2. Compute Degrees [cite: 173]
        # Dv: diagonal matrix of vertex degrees (row sum of H)
        # De: diagonal matrix of hyperedge degrees (col sum of H)
        dv = np.sum(H_inc, axis=1)
        de = np.sum(H_inc, axis=0)

        # Avoid division by zero
        dv[dv == 0] = 1e-10
        de[de == 0] = 1e-10

        Dv_inv_sqrt = np.diag(np.power(dv, -0.5))
        De_inv = np.diag(1.0 / de)

        # [cite_start]3. Compute Hypergraph Laplacian L = I - Dv^-0.5 * H * De^-1 * H^T * Dv^-0.5 [cite: 171]
        # L = D - S_hat (Typically normalized Laplacian in literature)
        S_hat = Dv_inv_sqrt @ H_inc @ De_inv @ H_inc.T @ Dv_inv_sqrt
        L = np.eye(n_samples) - S_hat

        # Ensure symmetry for numerical stability
        L = (L + L.T) / 2
        return L

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # --- 1. Data Preparation ---
        view_names = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in view_names])
        n, d = X_concat.shape

        # [cite_start]Y must be One-Hot encoded (n x c) [cite: 107]
        n_classes = len(np.unique(y))
        if y.ndim == 1:
            classes = np.unique(y)
            label_map = {label: i for i, label in enumerate(classes)}
            y_mapped = np.array([label_map[label] for label in y])
            Y = np.eye(len(classes))[y_mapped]
        else:
            Y = y
        c = Y.shape[1]

        # Rank r check
        rank_r = min(self.r, d, c)

        # --- 2. Hypergraph Laplacian Construction ---
        L = self._construct_hypergraph_laplacian(X_concat)

        # [cite_start]Centering Matrix H_c = I - 1/n * ee^T [cite: 198]
        # Used to center data (equivalent to removing bias b)
        H_c = np.eye(n) - np.ones((n, n)) / n

        # Precompute terms involving X and H_c
        X_c = H_c @ X_concat  # Centered X
        Y_c = H_c @ Y  # Centered Y

        # --- 3. Initialization ---
        # Initialize D (diagonal weight matrix for L2,1 norm)
        D_diag = np.ones(d)

        # Initialize A randomly
        A = np.random.randn(d, rank_r)

        # [cite_start]--- 4. Alternating Optimization [cite: 203-223] ---
        for iter_idx in range(self.max_iter):

            # --- Step 1: Update A (Generalized Eigenvalue Problem) ---
            # Maximize: Tr( A^T (X H H Y Y^T H H X^T) A ) / Tr( A^T (X H H X^T + alpha X L X^T + lambda D) A )
            # [cite_start]对应论文 Eq (19) 和 (20) [cite: 218-221]

            # Scatter matrices
            # Sb = X H Y Y^T H X^T (Numerator)
            # Sw = X H X^T + alpha X L X^T + lambda D (Denominator)

            # Computation optimization: X H H X^T = X_c^T X_c (since H H = H and X_c = H X)
            XtX_c = X_c.T @ X_c
            XtY_c = X_c.T @ Y_c

            Sb = XtY_c @ XtY_c.T

            # [cite_start]X L X^T term [cite: 177]
            XLXt = X_concat.T @ L @ X_concat

            Sw = XtX_c + self.alpha * XLXt + self.lambda_ * np.diag(D_diag)

            # Add epsilon for numerical stability
            Sw += 1e-6 * np.eye(d)

            # Solve Generalized Eigenvalue Problem: Sb v = w Sw v
            # [cite_start]We need largest eigenvalues [cite: 223]
            try:
                # scipy.linalg.eigh can handle generalized eigenvalue problems for symmetric/hermitian matrices
                # Sb and Sw are symmetric
                eigvals, eigvecs = scipy.linalg.eigh(Sb, Sw)
                # Select top-r eigenvectors (eigh returns eigenvalues in ascending order)
                A = eigvecs[:, -rank_r:]
            except (np.linalg.LinAlgError, ValueError):
                # Fallback if singular or convergence fails
                A = np.random.randn(d, rank_r)

            # --- Step 2: Update B ---
            # B = (A^T (Sw) A)^-1 A^T X H H Y
            # [cite_start]对应论文 Eq (16) 和 (17) [cite: 215]

            term_inv = A.T @ Sw @ A
            term_rhs = A.T @ XtY_c

            try:
                B = np.linalg.solve(term_inv, term_rhs)
            except np.linalg.LinAlgError:
                B = np.linalg.pinv(term_inv) @ term_rhs

            # --- Step 3: Update D ---
            # d_ii = 1 / (2 * ||w^i||_2) where W = AB
            # [cite_start]对应论文 Eq (13) [cite: 201]
            W = A @ B
            w_norms = np.linalg.norm(W, axis=1)
            D_diag = 0.5 / (w_norms + 1e-10)

        # --- 5. Feature Selection ---
        # [cite_start]Final W = AB [cite: 226]
        W_final = A @ B

        # Split W back to views for selection
        W_dict = {}
        curr = 0
        for name in view_names:
            dim = X[name].shape[1]
            W_dict[name] = W_final[curr:curr + dim]
            curr += dim

        selected = select_from_weights(W_dict, X, n)
        return FitResult(selected_features=selected)


# ==========================================
# Baseline 3: SCFS (Locally Sparse)
# ==========================================
class SCFS(BaseMethod):
    """
    Method: A supervised multi-view feature selection method based on locally sparse regularization and block computing
    Paper: Lin et al., Information Sciences 2022

    Implementation Key Points:
    1. Block Computing: Decomposes the problem by Class (c) and View (v).
    2. ADMM Solver: Solves sub-problems for each block W_vc.
    3. Locally Sparse Regularizer: L2,1 norm applied on each block W_vc.
    """

    def __init__(self, name="SCFS", lambda_=0.1, rho=1.0, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.lambda_ = lambda_  # Regularization parameter (对应论文中的参数 beta)
        self.rho = rho  # ADMM penalty parameter
        self.max_iter = max_iter

    def _solve_subproblem_W(self, X_v_c, P_c, U_v_c, rho):
        """
        求解 W_vc 子问题:
        min_W ||X_vc * W - (P_c - U_vc/rho)||_F^2 + (lambda/rho) * ||W||_2,1
        """
        # 目标变量 target
        target = P_c - U_v_c / rho

        # 1. 岭回归近似步 (Ridge Step)
        # (X'X + I)^-1 X'Y
        # 实际论文中为了加速可能使用了 SMW 公式，这里为稳定性使用标准求解
        d = X_v_c.shape[1]
        XtX = X_v_c.T @ X_v_c
        XtY = X_v_c.T @ target

        # 加入微小扰动保证可逆
        W_temp = np.linalg.solve(XtX + np.eye(d) * 1e-6, XtY)

        # 2. 软阈值步 (Proximal Step for L2,1)
        # 使用之前定义的辅助函数 soft_threshold_row
        threshold = self.lambda_ / rho
        W_new = soft_threshold_row(W_temp, threshold)

        return W_new

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # --- 1. 数据准备与分块 (Data Splitting) ---
        view_names = sorted(X.keys())
        n_views = len(view_names)
        n_samples = len(y)

        # 获取所有类别
        classes = np.unique(y)
        n_classes = len(classes)

        # 构建 One-Hot 标签矩阵 (用于分块回归的目标)
        # P_true: (n_samples, n_classes)
        label_map = {c: i for i, c in enumerate(classes)}
        Y_mat = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            Y_mat[i, label_map[label]] = 1

        # 记录各视图维度
        dims = {v: X[v].shape[1] for v in view_names}

        # --- 2. 变量初始化 ---
        # W: 嵌套字典 W[class_idx][view_name] -> shape (d_v, 1)
        # 针对每个类别 c，每个视图 v 都有一个权重向量 W_vc
        W = {c_idx: {v: np.zeros((dims[v], 1)) for v in view_names} for c_idx in range(n_classes)}

        # P: 共享回归目标 (Auxiliary Variable), P[class_idx] -> (n_samples, 1)
        # 初始化为真实标签
        P = {c_idx: Y_mat[:, c_idx:c_idx + 1].copy() for c_idx in range(n_classes)}

        # U: 拉格朗日乘子, U[class_idx][view_name] -> (n_samples, 1)
        U = {c_idx: {v: np.zeros((n_samples, 1)) for v in view_names} for c_idx in range(n_classes)}

        # --- 3. ADMM 主循环 (Block Computing) ---
        for iter_idx in range(self.max_iter):

            # 按类别遍历 (Block by Class)
            for c_idx in range(n_classes):
                Y_c = Y_mat[:, c_idx:c_idx + 1]

                # --- Step 3.1: 更新 W_vc (按视图并行) ---
                # 对应论文中求解 W 的子问题
                term_sum_for_P = np.zeros((n_samples, 1))  # 用于下一步更新 P

                for v in view_names:
                    X_v = X[v]
                    # 调用子问题求解器
                    W[c_idx][v] = self._solve_subproblem_W(
                        X_v, P[c_idx], U[c_idx][v], self.rho
                    )
                    # 累加用于 P 的更新: X_v * W_vc + U_vc/rho
                    term_sum_for_P += X_v @ W[c_idx][v] + U[c_idx][v] / self.rho

                # --- Step 3.2: 更新 P_c (共享目标) ---
                # P_c 是各视图预测结果的"共识"
                # P_c = (sum(XW + U/rho) + Y_c) / (N_views + 1)
                # 这一步将不同视图的信息融合
                P[c_idx] = (self.rho * term_sum_for_P + Y_c) / (self.rho * n_views + 1.0)

                # --- Step 3.3: 更新乘子 U_vc ---
                for v in view_names:
                    # U = U + rho * (XW - P)
                    resid = X[v] @ W[c_idx][v] - P[c_idx]
                    U[c_idx][v] += self.rho * resid

        # --- 4. 特征筛选 ---
        # 将各类别下的 W_vc 拼接起来，形成每个视图完整的权重矩阵 W_v
        # W_v shape: (d_v, n_classes)
        W_dict_final = {}
        for v in view_names:
            # 水平拼接该视图下所有类别的权重向量
            cols = [W[c_idx][v] for c_idx in range(n_classes)]
            W_dict_final[v] = np.hstack(cols)

        # 使用通用的 select_from_weights 进行打分和排序
        # 这里会自动计算 L2 norm (axis=1)，即衡量该特征对所有类别的综合贡献
        selected = select_from_weights(W_dict_final, X, n_samples)

        return FitResult(selected_features=selected)