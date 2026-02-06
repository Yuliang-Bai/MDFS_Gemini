import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from ...base import BaseMethod, FitResult


class AdaCoop(BaseMethod):
    """
    Baseline: AdaCoop (Adaptive Cooperative Learning)
    核心思想: 融合 Lasso 惩罚 (稀疏) 和 Agreement 惩罚 (协同)。
    代理实现: ElasticNet (L1 + L2) 是 AdaCoop 在线性回归下的近似形式。
    """

    def fit(self, X, y):
        # 1. 拼接所有模态
        X_concat = np.hstack(list(X.values()))

        # 2. 训练 ElasticNet
        # alpha=0.1, l1_ratio=0.5 模拟协同与稀疏的平衡
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        model.fit(X_concat, y)

        # 3. 筛选 (取绝对值最大的 Top-K)
        n = len(y)
        k_total = int(n / np.log(n))

        coefs = np.abs(model.coef_)
        top_idx = np.argsort(coefs)[::-1][:k_total]

        # 4. 映射回各模态
        selected = {}
        curr = 0
        top_set = set(top_idx)
        for name, data in X.items():
            dim = data.shape[1]
            # 检查当前模态区间内的索引是否被选中
            sel = [i - curr for i in top_idx if curr <= i < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class MSGLasso(BaseMethod):
    """
    Baseline: Multivariate Sparse Group Lasso
    核心思想: 组稀疏 (模态层面) + 组内稀疏 (特征层面)。
    代理实现: 
    1. 计算每个模态的整体重要性 (Group Norm)。
    2. 根据重要性分配筛选名额。
    3. 组内使用 Lasso 进行筛选。
    """

    def fit(self, X, y):
        n = len(y)
        k_total = int(n / np.log(n))

        # 1. 计算组重要性 (Group Importance)
        group_scores = {}
        for name, data in X.items():
            # 使用 Ridge 回归系数的 L2 范数作为组重要性近似
            lr = LinearRegression().fit(data, y)
            group_scores[name] = np.linalg.norm(lr.coef_)

        total_score = sum(group_scores.values()) + 1e-8

        selected = {}
        for name, data in X.items():
            # 2. 分配名额
            k_m = int(k_total * (group_scores[name] / total_score))
            k_m = max(1, k_m)

            # 3. 组内筛选 (Lasso)
            lasso = Lasso(alpha=0.05).fit(data, y)
            idx = np.argsort(np.abs(lasso.coef_))[::-1][:k_m]
            selected[name] = idx.tolist()

        return FitResult(selected_features=selected)


class SLRFS(BaseMethod):
    """
    Baseline: Sparse Low-Rank Feature Selection
    核心思想: 利用低秩矩阵分解捕捉相关性，同时进行稀疏选择。
    代理实现: PLS (Partial Least Squares) 提取低秩成分，根据 Loadings 筛选。
    """

    def fit(self, X, y):
        X_concat = np.hstack(list(X.values()))
        n = len(y)
        k_total = int(n / np.log(n))

        # 1. 低秩分解 (PLS with 5 components)
        pls = PLSRegression(n_components=5)
        pls.fit(X_concat, y)

        # 2. 计算特征重要性 (Sum of squared loadings)
        # x_loadings_: (n_features, n_components)
        importance = np.sum(pls.x_loadings_ ** 2, axis=1)

        # 3. 筛选
        top_idx = np.argsort(importance)[::-1][:k_total]

        selected = {}
        curr = 0
        for name, data in X.items():
            dim = data.shape[1]
            sel = [i - curr for i in top_idx if curr <= i < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)