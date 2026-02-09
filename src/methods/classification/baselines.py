import numpy as np
from typing import Dict, List, Optional
from math import floor, log
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph
from ...base import BaseMethod, FitResult


class BaseMultiViewLineMethod(BaseMethod):
    """线性方法基类"""

    def _rank_and_select(self, weights: np.ndarray, X_dict: Dict[str, np.ndarray], n_samples: int):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        # weights: (Total_Features, n_classes) or (Total_Features, )
        if weights.ndim > 1:
            scores = np.linalg.norm(weights, axis=1)  # L2 norm across classes
        else:
            scores = np.abs(weights)

        selected = {}
        current_idx = 0
        feature_names = sorted(X_dict.keys())
        total_dims = sum([v.shape[1] for v in X_dict.values()])

        for name in feature_names:
            dim = X_dict[name].shape[1]
            w_view = scores[current_idx: current_idx + dim]

            k_m = max(1, int(k_total * (dim / total_dims)))
            idx = np.argsort(w_view)[::-1][:k_m]
            selected[name] = idx.tolist()
            current_idx += dim
        return selected


class SCFS(BaseMultiViewLineMethod):
    """
    Method 1: SCFS (Sharing Multi-view Feature Selection via ADMM) [Lin et al., 2019]
    Objective: min_W ||XW - Y||_F^2 + lambda ||W||_2,1
    """

    def __init__(self, name="SCFS", alpha=0.1, rho=1.0, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        feature_names = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in feature_names])
        n, d = X_concat.shape
        # 处理二分类或多分类
        n_classes = len(np.unique(y))
        if n_classes == 2:
            # 二分类可以转为单列 -1/1 或 0/1
            Y_target = y.reshape(-1, 1).astype(float)
            Y_target[Y_target == 0] = -1  # 转换为 -1/1
        else:
            Y_target = np.eye(n_classes)[y]

        # ADMM 求解 Group Lasso
        W = np.zeros((d, Y_target.shape[1]))
        Z = np.zeros_like(W)
        U = np.zeros_like(W)

        XtX = X_concat.T @ X_concat
        XtY = X_concat.T @ Y_target
        I = np.eye(d)
        inv_mat = np.linalg.inv(XtX + self.rho * I)

        for _ in range(self.max_iter):
            W = inv_mat @ (XtY + self.rho * (Z - U))
            # Soft Thresholding for L2,1
            V = W + U
            v_norms = np.linalg.norm(V, axis=1, keepdims=True)
            scale = np.maximum(0, 1 - self.alpha / (self.rho * v_norms + 1e-10))
            Z = V * scale
            U = U + W - Z

        return FitResult(selected_features=self._rank_and_select(Z, X, n))


class HLRFS(BaseMultiViewLineMethod):
    """
    Method 2: HLRFS (Hypergraph Low-Rank Feature Selection) [Cheng et al., 2017]
    Approximation: Graph Regularized L2,1 Selection
    Objective: min_W ||XW - Y||^2 + beta * Tr(W'XLX'W) + gamma * ||W||_2,1
    """

    def __init__(self, name="HLRFS", beta=1.0, gamma=0.1, n_neighbors=5, **kwargs):
        super().__init__(name, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        feature_names = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in feature_names])
        n, d = X_concat.shape
        n_classes = len(np.unique(y))
        Y_target = np.eye(n_classes)[y] if n_classes > 2 else y.reshape(-1, 1)

        # Build Graph Laplacian
        A = kneighbors_graph(X_concat, self.n_neighbors, mode='connectivity', include_self=False)
        L = csgraph.laplacian(A, normed=False).toarray()

        # Iterative Solver (Reweighted Least Squares)
        W = np.random.randn(d, Y_target.shape[1]) * 0.01
        XtX = X_concat.T @ X_concat
        XLX = X_concat.T @ L @ X_concat
        XtY = X_concat.T @ Y_target

        for _ in range(15):  # Fast iter
            w_norms = np.linalg.norm(W, axis=1)
            D_diag = 0.5 / (w_norms + 1e-8)
            # System: (XtX + beta*XLX + gamma*D) W = XtY
            # Add small jitter to diagonal for stability
            A_sys = XtX + self.beta * XLX + np.diag(self.gamma * D_diag) + 1e-6 * np.eye(d)
            try:
                W = np.linalg.solve(A_sys, XtY)
            except:
                break

        return FitResult(selected_features=self._rank_and_select(W, X, n))


class LSRFS(BaseMultiViewLineMethod):
    """
    Method 3: LSRFS (Locally Sparse Regularization) [Lin et al., 2022]
    Core Idea: Feature selection with 'local' sparsity emphasis.
    Implementation: Approximated as Elastic Net style or Reweighted L1/L2 scheme on View Blocks.
    Here we implement a standard L2,1 + L1 regularized Least Squares to capture both global row sparsity
    and local entry sparsity, which is the spirit of 'Locally Sparse'.
    """

    def __init__(self, name="LSRFS", lambda1=0.1, lambda2=0.1, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.lambda1 = lambda1  # Controls L2,1 (Group/Row sparsity)
        self.lambda2 = lambda2  # Controls L1 (Local sparsity)
        self.max_iter = max_iter

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # PGD (Proximal Gradient Descent) solver
        feature_names = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in feature_names])
        n, d = X_concat.shape
        n_classes = len(np.unique(y))
        Y_target = np.eye(n_classes)[y] if n_classes > 2 else y.reshape(-1, 1)

        W = np.zeros((d, Y_target.shape[1]))
        lr = 1e-3

        for _ in range(self.max_iter):
            # Gradient of Data Term: X.T(XW - Y)
            grad = X_concat.T @ (X_concat @ W - Y_target)

            # Gradient Descent
            W_temp = W - lr * grad

            # Proximal Operator for L1 (Local Sparse)
            # Soft Thresholding element-wise
            W_temp = np.sign(W_temp) * np.maximum(np.abs(W_temp) - lr * self.lambda2, 0)

            # Proximal Operator for L2,1 (Group Sparse)
            # Block Soft Thresholding row-wise
            row_norms = np.linalg.norm(W_temp, axis=1, keepdims=True)
            # Avoid division by zero
            scale = np.maximum(0, 1 - (lr * self.lambda1) / (row_norms + 1e-8))
            W = W_temp * scale

        return FitResult(selected_features=self._rank_and_select(W, X, n))