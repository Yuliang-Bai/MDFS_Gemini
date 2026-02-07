import numpy as np
from typing import Dict, List, Optional, Tuple
from ...base import BaseMethod, FitResult


class MDFSRegressor(BaseMethod):
    """
    Multimodal Deep Feature Screening (Regression)
    核心机制: Gated Autoencoder + Entropy/Sparsity Regularization + MRDC Screening
    """

    def __init__(self, name="MDFS_Reg",
                 latent_dim=10,
                 hidden_ratio=0.5,
                 temperature=0.7,
                 lambda_r=1.0,
                 lambda_ent=0.05,
                 lambda_sp=0.05,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.latent_dim = latent_dim
        self.hidden_ratio = hidden_ratio
        self.temperature = temperature
        self.lambda_r = lambda_r
        self.lambda_ent = lambda_ent
        self.lambda_sp = lambda_sp
        self.epsilon = 1e-8

    def _pca_encode(self, x: np.ndarray, dim: int):
        """模拟线性自编码器 (Encoder + Decoder)"""
        # Center
        mu = x.mean(axis=0)
        xc = x - mu
        # SVD
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        # Encode: z = x @ V.T
        z = xc @ vt[:dim].T
        # Decode: x_hat = z @ V
        x_hat = z @ vt[:dim] + mu
        return z, x_hat

    def _get_rank(self, x: np.ndarray) -> np.ndarray:
        """计算向量的秩 (用于 MRDC)"""
        return np.argsort(np.argsort(x)) + 1

    def _mrdc_proxy(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        多元秩距离相关 (MRDC) 的快速代理计算。
        注：完整的 MRDC 需要计算距离矩阵的距离协方差。
        为了模拟效率，这里计算 x 与 z 的第一主成分的秩相关，
        或者计算 x 与 z 所有维度的平均秩相关。
        这里采用：x 与 z 的 CCA (典型相关) 或 平均相关作为筛选依据。
        """
        # 简单高效实现：计算 x (rank) 与 z (rank) 每一维的相关系数取最大值
        rx = self._get_rank(x)
        corrs = []
        for j in range(z.shape[1]):
            rz = self._get_rank(z[:, j])
            corrs.append(np.abs(np.corrcoef(rx, rz)[0, 1]))
        return np.max(corrs)  # 取与 z 任何一维的最大相关性

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # 1. 预处理 (Standardize)
        X_std = {}
        for k, v in X.items():
            X_std[k] = (v - v.mean(axis=0)) / (v.std(axis=0) + self.epsilon)

        # 2. 独立编码 (Phi_m) & 重构损失 (L_r)
        reps = []
        loss_r = 0
        for name, data in X_std.items():
            h_dim = max(1, int(data.shape[1] * self.hidden_ratio))
            r, x_hat = self._pca_encode(data, h_dim)
            reps.append(r)
            loss_r += np.mean((data - x_hat) ** 2)
        loss_r /= len(X)

        # 3. 门控网络 (Gating)
        # 模拟：利用 r 对 y 的预测能力来初始化门控 logits
        scores = []
        for r in reps:
            # 简单线性回归 r -> y 的拟合度 (R2 or inverse MSE)
            coef = np.linalg.lstsq(r, y - y.mean(), rcond=None)[0]
            mse = np.mean((y - y.mean() - r @ coef) ** 2)
            scores.append(1.0 / (mse + self.epsilon))

        scores = np.array(scores)
        # Softmax (Eq. 1)
        logits = np.log(scores + self.epsilon) / self.temperature
        alpha = np.exp(logits - np.max(logits))
        alpha = alpha / np.sum(alpha)

        # 4. 正则化损失 (L_ent, L_sp)
        loss_ent = -np.sum(alpha * np.log(alpha + self.epsilon))
        loss_sp = np.sum(alpha ** 2)

        # 5. 联合表示 (Joint Rep)
        # Weighted Concatenation
        z_joint = np.hstack([r * w for r, w in zip(reps, alpha)])
        # Compress to z
        z, _ = self._pca_encode(z_joint, self.latent_dim)

        # 6. Min-Max 归一化 z (用于筛选)
        z_encode = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + self.epsilon)

        # 7. 特征筛选 (Feature Screening)
        n = len(y)
        k_total = int(n / np.log(n))

        selected = {}
        total_dims = sum(d.shape[1] for d in X.values())

        for name, data in X.items():
            # 分配筛选数量
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))

            # 计算每个特征的重要性 (MRDC with z_encode)
            importances = []
            for j in range(data.shape[1]):
                imp = self._mrdc_proxy(data[:, j], z_encode)
                importances.append(imp)

            # 排序筛选
            idx = np.argsort(importances)[::-1][:k_m]
            selected[name] = idx.tolist()

        return FitResult(
            selected_features=selected,
            model_state={
                "alpha": alpha,
                "losses": {"r": loss_r, "ent": loss_ent, "sp": loss_sp}
            }
        )