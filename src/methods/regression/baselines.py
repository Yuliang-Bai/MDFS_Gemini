import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.cross_decomposition import PLSRegression
from ...base import BaseMethod, FitResult


class AdaCoop(BaseMethod):
    """
    Baseline: AdaCoop (Adaptive Cooperative Learning)
    """

    def __init__(self, name="AdaCoop", **kwargs):
        super().__init__(name=name, **kwargs)

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        # 1. 确保顺序一致
        sorted_keys = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in sorted_keys])

        # 2. 训练 ElasticNetCV
        # 【关键修改】n_jobs=1，避免与外层多进程冲突
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            random_state=42,
            n_jobs=1,  # 修改此处：从 -1 改为 1
            max_iter=2000
        )
        model.fit(X_concat, y)

        # 3. 筛选
        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1

        coefs = np.abs(model.coef_)
        top_idx = np.argsort(coefs)[::-1][:k_total]

        # 4. 映射回各模态
        selected = {}
        curr = 0
        for name in sorted_keys:
            data = X[name]
            dim = data.shape[1]
            sel = [i - curr for i in top_idx if curr <= i < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class MSGLasso(BaseMethod):
    """
    Baseline: Multivariate Sparse Group Lasso
    """

    def __init__(self, name="MSGLasso", **kwargs):
        super().__init__(name=name, **kwargs)

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1

        # 1. 计算组重要性 (使用 Ridge)
        group_scores = {}
        for name, data in X.items():
            ridge = Ridge(alpha=1.0).fit(data, y)
            group_scores[name] = np.linalg.norm(ridge.coef_)

        total_score = sum(group_scores.values()) + 1e-8

        selected = {}
        for name, data in X.items():
            # 2. 分配名额
            ratio = group_scores[name] / total_score
            k_m = int(k_total * ratio)
            k_m = max(1, k_m)

            # 3. 组内筛选 (使用 LassoCV)
            # 【关键修改】n_jobs=1，避免与外层多进程冲突
            lasso = LassoCV(
                cv=3,
                random_state=42,
                n_jobs=1,  # 修改此处：从 -1 改为 1
                max_iter=2000
            ).fit(data, y)

            idx = np.argsort(np.abs(lasso.coef_))[::-1][:k_m]
            selected[name] = idx.tolist()

        return FitResult(selected_features=selected)


class SLRFS(BaseMethod):
    """
    Baseline: Sparse Low-Rank Feature Selection
    """

    def __init__(self, name="SLRFS", **kwargs):
        super().__init__(name=name, **kwargs)

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        sorted_keys = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in sorted_keys])

        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1

        # 1. 低秩分解
        n_components = min(5, X_concat.shape[1], X_concat.shape[0] - 1)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_concat, y)

        # 2. 计算特征重要性
        importance = np.sum(pls.x_loadings_ ** 2, axis=1)

        # 3. 筛选
        top_idx = np.argsort(importance)[::-1][:k_total]

        selected = {}
        curr = 0
        for name in sorted_keys:
            data = X[name]
            dim = data.shape[1]
            sel = [i - curr for i in top_idx if curr <= i < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)