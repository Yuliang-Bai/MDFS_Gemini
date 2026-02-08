import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.cross_decomposition import PLSRegression
from ...base import BaseMethod, FitResult


class AdaCoop(BaseMethod):
    """
    Implementation of Cooperative Learning (Algorithm 1) from Ding et al. (2022).
    This constructs an augmented matrix to enforce agreement between views.
    """

    def __init__(self, name="AdaCoop", rho_grid=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # 默认搜索的 rho 值，包含 0 (Early Fusion), 1 (Late Fusion 近似), 以及其他强度
        #  建议在一个 grid 上搜索 rho
        self.rho_grid = rho_grid if rho_grid is not None else [0, 0.1, 0.5, 1.0, 5.0, 10.0]

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        # 1. 预处理：确保只有两个视图 (论文主要推导基于双视图 X 和 Z [cite: 106])
        sorted_keys = sorted(X.keys())
        if len(sorted_keys) != 2:
            # 如果超过2个视图，论文建议使用迭代算法或成对约束 [cite: 583]，
            # 这里为保持代码简洁，暂时只处理前两个视图，实际应用需扩展
            print(f"Warning: AdaCoop expects 2 views, found {len(sorted_keys)}. Using first two.")
            sorted_keys = sorted_keys[:2]

        view1_name, view2_name = sorted_keys
        X1 = X[view1_name]
        X2 = X[view2_name]

        # 标准化是必须的，假设输入的X和Z列已经标准化 [cite: 131]
        # 这里为了安全再次进行标准化处理
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        X1_std = scaler1.fit_transform(X1)
        X2_std = scaler2.fit_transform(X2)

        n_samples = y.shape[0]
        p1 = X1_std.shape[1]
        p2 = X2_std.shape[1]

        # 2. 交叉验证选择最佳 rho [cite: 170]
        # Remark A: CV 必须基于原始 X/Z 的行进行划分，而不是增广后的矩阵
        best_rho = 0
        best_mse = float('inf')

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for rho in self.rho_grid:
            cv_errors = []

            for train_idx, val_idx in kf.split(X1_std):
                # 切分数据
                X1_train, X1_val = X1_std[train_idx], X1_std[val_idx]
                X2_train, X2_val = X2_std[train_idx], X2_std[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 构造增广矩阵 (Augmented Data) 用于训练 [cite: 148, 168]
                # X_tilde = [ X      Z     ]
                #           [ -sq_r*X sq_r*Z]
                # y_tilde = [ y ]
                #           [ 0 ]
                sqrt_rho = np.sqrt(rho)
                n_train = len(y_train)

                # 上半部分: 预测误差项
                top_X = np.hstack([X1_train, X2_train])
                top_y = y_train

                # 下半部分: 一致性惩罚项 (Agreement Penalty)
                # 只有 rho > 0 时才需要添加约束行
                if rho > 0:
                    bottom_X = np.hstack([-sqrt_rho * X1_train, sqrt_rho * X2_train])
                    bottom_y = np.zeros(n_train)

                    X_aug = np.vstack([top_X, bottom_X])
                    y_aug = np.concatenate([top_y, bottom_y])
                else:
                    X_aug = top_X
                    y_aug = top_y

                # 使用 Lasso/ElasticNet 求解问题 [cite: 150]
                # 这里使用固定的小 alpha 或简单的 CV 来加速内部循环
                model = Lasso(alpha=0.1, max_iter=2000)  # 简化：实际应嵌套 CV 选 lambda
                model.fit(X_aug, y_aug)

                # 预测：在验证集上不使用增广矩阵，直接预测 [cite: 137]
                # f(X) + f(Z) = X_beta_x + Z_beta_z
                X_val_concat = np.hstack([X1_val, X2_val])
                y_pred = model.predict(X_val_concat)

                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)

            avg_mse = np.mean(cv_errors)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_rho = rho

        # 3. 使用最佳 rho 在全部数据上重新训练 [cite: 170]
        sqrt_rho = np.sqrt(best_rho)

        top_X = np.hstack([X1_std, X2_std])
        top_y = y

        if best_rho > 0:
            bottom_X = np.hstack([-sqrt_rho * X1_std, sqrt_rho * X2_std])
            bottom_y = np.zeros(n_samples)
            X_final = np.vstack([top_X, bottom_X])
            y_final = np.concatenate([top_y, bottom_y])
        else:
            X_final = top_X
            y_final = top_y

        # 最终模型训练，这里可以使用 ElasticNetCV 来自动选择 lambda [cite: 153]
        final_model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=5,
            random_state=42,
            n_jobs=1,
            max_iter=3000
        )
        final_model.fit(X_final, y_final)

        # 4. 特征筛选逻辑 (保持与原代码一致)
        # 系数被拼接在一起了： [theta_x, theta_z]
        coefs = np.abs(final_model.coef_)

        # 简单的 Top-K 筛选策略
        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1
        top_idx = np.argsort(coefs)[::-1][:k_total]

        selected = {}
        curr = 0

        # 映射回原始视图 (X 和 Z)
        # 注意顺序：sorted_keys = [view1, view2]
        dims = [p1, p2]
        for i, name in enumerate(sorted_keys):
            dim = dims[i]
            # 找到属于当前视图的特征索引
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
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