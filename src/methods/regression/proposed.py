import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import rankdata
import dcor
from math import floor, log
from ...base import BaseMethod, FitResult


# ==========================================
# 1. 设备与辅助函数
# ==========================================
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


DEVICE = get_device()


class MLPBlock(nn.Module):
    """
    通用 MLP 模块：支持指定层数，默认为 Linear -> ReLU -> ...
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_d = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim

        # 最后一层
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. MDFS 核心类 (Regression)
# ==========================================
class MDFSRegressor(BaseMethod):
    """
    MDFS (Multimodal Deep Feature Screening) for Regression.

    Structure:
    - Encoders/Decoders: 3-layer MLP
    - Gating/Compress/Predict: 2-layer MLP
    - Screening: Marginal ECDF + Distance Correlation
    """

    def __init__(self, name="MDFS_Reg",
                 # 网络维度参数
                 latent_dim: int = 16,  # 最终联合表示 Z 的维度 (h)
                 view_latent_dim: int = 16,  # 每个模态的中间表示维度 (d_m)
                 hidden_dim: int = 64,  # 网络隐藏层节点数
                 dropout: float = 0.0,

                 # 训练参数
                 lr: float = 1e-3,
                 epochs: int = 100,  # 模拟通常数据量小，epoch可适当增加
                 batch_size: int = 32,
                 temperature: float = 0.7,

                 # 损失权重
                 lambda_r: float = 1.0,  # 重构
                 lambda_ent: float = 0.05,  # 熵
                 lambda_sp: float = 0.05,  # 稀疏 (二范数)

                 **kwargs):
        super().__init__(name, **kwargs)
        self.latent_dim = latent_dim
        self.view_latent_dim = view_latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature

        self.lambda_r = lambda_r
        self.lambda_ent = lambda_ent
        self.lambda_sp = lambda_sp
        self.epsilon = 1e-8

        # 模型组件 (在 fit 时初始化)
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.gating_net = None
        self.compress_net = None
        self.predict_net = None

    def _init_networks(self, input_dims: Dict[str, int], device: str):
        """根据输入数据维度初始化所有子网络"""
        self.modality_names = sorted(input_dims.keys())
        total_view_latent = len(self.modality_names) * self.view_latent_dim

        for name in self.modality_names:
            p_m = input_dims[name]
            # 编码器: 3层 (Input -> Hidden -> Hidden -> ViewLatent)
            self.encoders[name] = MLPBlock(p_m, self.view_latent_dim, self.hidden_dim, num_layers=3,
                                           dropout=self.dropout).to(device)
            # 解码器: 3层 (ViewLatent -> Hidden -> Hidden -> Output)
            self.decoders[name] = MLPBlock(self.view_latent_dim, p_m, self.hidden_dim, num_layers=3,
                                           dropout=self.dropout).to(device)

        # 门控网络: 2层 (Concat(r_m) -> Hidden -> M)
        self.gating_net = MLPBlock(total_view_latent, len(self.modality_names), self.hidden_dim, num_layers=2).to(
            device)

        # 压缩网络: 2层 (WeightedConcat(r_m) -> Hidden -> Z)
        self.compress_net = MLPBlock(total_view_latent, self.latent_dim, self.hidden_dim, num_layers=2).to(device)

        # 预测网络: 2层 (Z -> Hidden -> Y)
        self.predict_net = MLPBlock(self.latent_dim, 1, self.hidden_dim, num_layers=2).to(device)

    def _marginal_ecdf_map(self, A: np.ndarray) -> np.ndarray:
        """
        边际经验分布函数转化 (Marginal ECDF Map)
        """
        if A.ndim == 1:
            A = A[:, None]
        n_local, d_local = A.shape
        out = np.empty((n_local, d_local), dtype=float)
        for i in range(d_local):
            # method="max" 对应最大秩，除以 n 归一化到 (0, 1]
            out[:, i] = rankdata(A[:, i], method="max") / n_local
        return out

    def _calculate_feature_scores(self, X_modality: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        计算特征重要性分数：基于 ECDF + dCor
        """
        n_local, p_local = X_modality.shape

        # 1. 转换 Z (联合表示)
        Zt = self._marginal_ecdf_map(Z)

        # 2. 转换 X (当前模态原始特征)
        Xt = self._marginal_ecdf_map(X_modality)

        sc = np.zeros(p_local, dtype=float)

        # 3. 逐特征计算距离相关系数
        for j in range(p_local):
            # dcor.distance_correlation 需要 (N,) 或 (N, D)
            # Xt[:, j] 是 (N,)
            sc[j] = dcor.distance_correlation(Xt[:, j], Zt)

        return sc

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # ==========================
        # 1. 数据准备 (Data Prep)
        # ==========================
        input_dims = {k: v.shape[1] for k, v in X.items()}
        n_samples = len(y)

        # 初始化网络
        self._init_networks(input_dims, DEVICE)

        # 转换为 Tensor
        X_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in X.items()}
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)

        # 优化器
        # 收集所有参数
        params = (list(self.encoders.parameters()) + list(self.decoders.parameters()) +
                  list(self.gating_net.parameters()) + list(self.compress_net.parameters()) +
                  list(self.predict_net.parameters()))
        optimizer = optim.Adam(params, lr=self.lr)
        criterion_mse = nn.MSELoss()

        # ==========================
        # 2. 训练循环 (Training Loop)
        # ==========================
        self.encoders.train()
        self.decoders.train()

        # 简单起见，这里使用全量 Batch (如果数据量大需改为 DataLoader)
        # 模拟数据通常 n=200-400，全量即可

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # --- Forward Pass ---

            # A. 独立编码与重构
            rs = []
            recon_loss = 0.0
            for name in self.modality_names:
                x_in = X_t[name]
                r_m = self.encoders[name](x_in)  # Encoder
                x_rec = self.decoders[name](r_m)  # Decoder
                rs.append(r_m)
                recon_loss += criterion_mse(x_rec, x_in)

            recon_loss /= len(self.modality_names)

            # 拼接所有 r_m: (N, M * d_m)
            r_concat = torch.cat(rs, dim=1)

            # B. 门控网络
            # logits: (N, M)
            gate_logits = self.gating_net(r_concat)
            # Softmax with temperature (Eq. 1)
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)

            # 正则化
            # L_ent = - sum(alpha * log(alpha))
            loss_ent = -torch.sum(alpha * torch.log(alpha + self.epsilon), dim=1).mean()
            # L_sp = sum(alpha^2)
            loss_sp = torch.sum(alpha ** 2, dim=1).mean()

            # C. 加权融合与压缩
            # alpha: (N, M), r_m: (N, d_m)
            # 扩展 alpha 以便进行元素级乘法
            weighted_rs = []
            for i, r_m in enumerate(rs):
                w = alpha[:, i:i + 1]  # (N, 1)
                weighted_rs.append(r_m * w)

            R_weighted = torch.cat(weighted_rs, dim=1)
            z = self.compress_net(R_weighted)  # (N, h)

            # D. 预测
            y_pred = self.predict_net(z)
            loss_s = criterion_mse(y_pred, y_t)

            # E. 总损失
            total_loss = (loss_s +
                          self.lambda_r * recon_loss +
                          self.lambda_ent * loss_ent +
                          self.lambda_sp * loss_sp)

            # --- Backward ---
            total_loss.backward()
            optimizer.step()

        # ==========================
        # 3. 特征筛选 (Inference & Screening)
        # ==========================
        self.encoders.eval()
        self.compress_net.eval()
        self.gating_net.eval()

        with torch.no_grad():
            # 重新前向传播一次获取最终 Z
            rs = [self.encoders[name](X_t[name]) for name in self.modality_names]
            r_concat = torch.cat(rs, dim=1)
            gate_logits = self.gating_net(r_concat)
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)
            weighted_rs = [rs[i] * alpha[:, i:i + 1] for i in range(len(rs))]
            R_weighted = torch.cat(weighted_rs, dim=1)
            z_final_t = self.compress_net(R_weighted)

            # 转回 Numpy
            Z_np = z_final_t.cpu().numpy()
            alpha_np = alpha.cpu().numpy()

        # 计算筛选
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(input_dims.values())
        selected = {}

        # 先计算所有模态的分数
        all_scores = {}
        for name, data in X.items():
            scores = self._calculate_feature_scores(data, Z_np)
            all_scores[name] = scores

        # 排序并截断
        for name, data in X.items():
            k_m = max(1, int(k_total * (input_dims[name] / total_dims)))
            scores = all_scores[name]
            # argsort 默认升序，取负号变成降序
            idx = np.argsort(-scores)[:k_m]
            selected[name] = idx.tolist()

        return FitResult(
            selected_features=selected,
            model_state={
                "alpha": alpha_np,
                "z": Z_np
            }
        )