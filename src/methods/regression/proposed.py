import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Optional
from scipy.stats import rankdata
import dcor
from math import floor, log
from ...base import BaseMethod, FitResult


# ==========================================
# 1. 设备与基础模块
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
    通用多层感知机模块 (General Multi-Layer Perceptron Block)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        in_d = input_dim

        # 构建隐藏层 (Hidden Layers)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h_dim

        # 构建输出层 (Output Layer)
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. MDFS 核心类 (Regression)
# ==========================================
class MDFSRegressor(BaseMethod):
    """
    MDFS: Multimodal Deep Feature Screening for Regression.

    Architecture:
      1. Heterogeneous Encoders: X_m -> [MLP_m] -> r_m
      2. Gating Network: [r_1..r_M] -> [MLP] -> alpha
      3. Joint Compression: WeightedConcat(r_m) -> [MLP] -> z
      4. Shared Decoding: z -> [MLP_m] -> X_rec_m
      5. Prediction: z -> [MLP] -> y
    """

    def __init__(self, name="MDFS_Reg",
                 # --- 维度定义 (Dimensions) ---
                 latent_dim: int = 16,  # 联合隐变量 z 的维度 (Joint Latent Dim)
                 view_latent_dim: int = 16,  # 模态隐变量 r 的维度 (View Latent Dim)

                 # --- 网络拓扑 (Network Topology) ---
                 # 支持两种格式：
                 # 1. 单层列表 [100, 50] -> 所有模态共用此结构
                 # 2. 双层列表 [[100, 50], [64]] -> 按模态顺序分别指定结构
                 encoder_struct: Union[List[int], List[List[int]]] = [64, 32],
                 decoder_struct: Union[List[int], List[List[int]]] = [32, 64],

                 dropout: float = 0.0,

                 # --- 训练参数 (Training Params) ---
                 lr: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int = 32,
                 temperature: float = 0.7,

                 # --- 正则化权重 (Regularization Weights) ---
                 lambda_r: float = 1.0,  # Reconstruction
                 lambda_ent: float = 0.05,  # Entropy
                 lambda_sp: float = 0.05,  # Sparsity

                 **kwargs):
        super().__init__(name, **kwargs)
        self.latent_dim = latent_dim
        self.view_latent_dim = view_latent_dim
        self.encoder_struct = encoder_struct
        self.decoder_struct = decoder_struct
        self.dropout = dropout

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature

        self.lambda_r = lambda_r
        self.lambda_ent = lambda_ent
        self.lambda_sp = lambda_sp
        self.epsilon = 1e-8

        # 组件容器
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.gating_net = None
        self.compress_net = None
        self.predict_net = None

    def _parse_struct_config(self, struct_config, num_views):
        """解析网络结构配置，确保其为双层列表格式"""
        # 情况1: 空列表或None -> 只有一层直接输出
        if not struct_config:
            return [[] for _ in range(num_views)]

        # 情况2: 双层列表 [[100], [50]] -> 针对每个模态
        if isinstance(struct_config[0], list):
            if len(struct_config) != num_views:
                raise ValueError(f"Config Error: Provided {len(struct_config)} structures for {num_views} views.")
            return struct_config

        # 情况3: 单层列表 [100, 50] -> 广播到所有模态
        return [struct_config] * num_views

    def _init_networks(self, input_dims: Dict[str, int], device: str):
        """初始化所有子网络"""
        self.modality_names = sorted(input_dims.keys())
        num_views = len(self.modality_names)

        # 1. 解析异构网络结构配置
        enc_structs = self._parse_struct_config(self.encoder_struct, num_views)
        dec_structs = self._parse_struct_config(self.decoder_struct, num_views)

        total_view_latent = num_views * self.view_latent_dim

        for i, name in enumerate(self.modality_names):
            p_m = input_dims[name]

            # Encoder: Input(p_m) -> [Specific Struct] -> ViewLatent
            self.encoders[name] = MLPBlock(
                input_dim=p_m,
                output_dim=self.view_latent_dim,
                hidden_layers=enc_structs[i],
                dropout=self.dropout
            ).to(device)

            # Decoder: JointLatent -> [Specific Struct] -> Output(p_m)
            self.decoders[name] = MLPBlock(
                input_dim=self.latent_dim,
                output_dim=p_m,
                hidden_layers=dec_structs[i],
                dropout=self.dropout
            ).to(device)

        # Gating Net: (Concat r) -> [64] -> alpha
        self.gating_net = MLPBlock(
            input_dim=total_view_latent,
            output_dim=num_views,
            hidden_layers=[64],
            dropout=self.dropout
        ).to(device)

        # Compress Net: (Weighted Concat r) -> [64] -> z
        self.compress_net = MLPBlock(
            input_dim=total_view_latent,
            output_dim=self.latent_dim,
            hidden_layers=[64],
            dropout=self.dropout
        ).to(device)

        # Predict Net: z -> [32] -> y
        self.predict_net = MLPBlock(
            input_dim=self.latent_dim,
            output_dim=1,
            hidden_layers=[32],
            dropout=self.dropout
        ).to(device)

    def _marginal_ecdf_map(self, A: np.ndarray) -> np.ndarray:
        """边际经验分布映射"""
        if A.ndim == 1:
            A = A[:, None]
        n_local, d_local = A.shape
        out = np.empty((n_local, d_local), dtype=float)
        for i in range(d_local):
            out[:, i] = rankdata(A[:, i], method="max") / n_local
        return out

    def _calculate_feature_scores(self, X_modality: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """计算特征筛选分数 (Distance Correlation)"""
        n_local, p_local = X_modality.shape

        # 安全性检查：如果 Z 是常数，dcor 会报错
        if np.var(Z) < 1e-9:
            return np.zeros(p_local)

        Zt = self._marginal_ecdf_map(Z)
        Xt = self._marginal_ecdf_map(X_modality)

        sc = np.zeros(p_local, dtype=float)
        for j in range(p_local):
            # 安全性检查：如果特征是常数，跳过
            if np.var(Xt[:, j]) < 1e-9:
                sc[j] = 0.0
                continue

            try:
                sc[j] = dcor.distance_correlation(Xt[:, j], Zt)
            except Exception:
                # 兜底防止崩溃
                sc[j] = 0.0
        return sc

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        # ==========================
        # 1. 初始化
        # ==========================
        input_dims = {k: v.shape[1] for k, v in X.items()}
        n_samples = len(y)

        self._init_networks(input_dims, DEVICE)

        # 数据转 Tensor
        X_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in X.items()}
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)

        # 优化器参数收集
        params = (list(self.encoders.parameters()) + list(self.decoders.parameters()) +
                  list(self.gating_net.parameters()) + list(self.compress_net.parameters()) +
                  list(self.predict_net.parameters()))

        if not params:
            raise ValueError("No parameters to optimize! Check network initialization.")

        optimizer = optim.Adam(params, lr=self.lr)
        criterion_mse = nn.MSELoss()

        # ==========================
        # 2. 训练循环
        # ==========================
        self.encoders.train()
        self.decoders.train()
        self.gating_net.train()
        self.compress_net.train()
        self.predict_net.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # --- Encoding ---
            rs = []
            for name in self.modality_names:
                r_m = self.encoders[name](X_t[name])
                rs.append(r_m)

            # --- Gating ---
            r_concat = torch.cat(rs, dim=1)
            gate_logits = self.gating_net(r_concat)
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)

            # --- Fusion ---
            weighted_rs = [r_m * alpha[:, i:i + 1] for i, r_m in enumerate(rs)]
            R_weighted = torch.cat(weighted_rs, dim=1)
            z = self.compress_net(R_weighted)

            # --- Reconstruction (Decoded from Z) ---
            recon_loss = 0.0
            for name in self.modality_names:
                x_rec = self.decoders[name](z)
                recon_loss += criterion_mse(x_rec, X_t[name])
            recon_loss /= len(self.modality_names)

            # --- Prediction ---
            y_pred = self.predict_net(z)
            loss_s = criterion_mse(y_pred, y_t)

            # --- Regularization ---
            loss_ent = -torch.sum(alpha * torch.log(alpha + self.epsilon), dim=1).mean()
            loss_sp = torch.sum(alpha ** 2, dim=1).mean()

            total_loss = (loss_s +
                          self.lambda_r * recon_loss +
                          self.lambda_ent * loss_ent +
                          self.lambda_sp * loss_sp)

            total_loss.backward()
            optimizer.step()

        # ==========================
        # 3. 筛选 (Inference)
        # ==========================
        self.encoders.eval()
        self.gating_net.eval()
        self.compress_net.eval()

        with torch.no_grad():
            rs = [self.encoders[name](X_t[name]) for name in self.modality_names]
            r_concat = torch.cat(rs, dim=1)
            gate_logits = self.gating_net(r_concat)
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)
            weighted_rs = [rs[i] * alpha[:, i:i + 1] for i in range(len(rs))]
            R_weighted = torch.cat(weighted_rs, dim=1)
            z_final = self.compress_net(R_weighted)

            Z_np = z_final.cpu().numpy()
            alpha_np = alpha.cpu().numpy()

        # 筛选计算
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(input_dims.values())
        selected = {}

        all_scores = {}
        for name, data in X.items():
            scores = self._calculate_feature_scores(data, Z_np)
            all_scores[name] = scores

        for name, data in X.items():
            k_m = max(1, int(k_total * (input_dims[name] / total_dims)))
            scores = all_scores[name]
            # argsort 默认升序，取负数变降序
            idx = np.argsort(-scores)[:k_m]
            selected[name] = idx.tolist()

        return FitResult(
            selected_features=selected,
            model_state={"alpha": alpha_np, "z": Z_np}
        )