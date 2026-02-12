import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Optional, Any
from scipy.stats import rankdata
import dcor
from math import floor, log
from ...base import BaseMethod, FitResult
from ...utils.metrics import calculate_selection_metrics


# ==========================================
# 1. 设备与基础模块
# ==========================================
def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    try:
        if torch.backends.mps.is_available(): return "mps"
    except AttributeError:
        pass
    return "cpu"


DEVICE = get_device()


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            in_d = h_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. MDFS 核心类 (Regression)
# ==========================================
class MDFSRegressor(BaseMethod):
    def __init__(self, name="MDFS_Reg",
                 latent_dim: int = 16, view_latent_dim: int = 16,
                 encoder_struct: Union[List[int], List[List[int]]] = [64, 32],
                 decoder_struct: Union[List[int], List[List[int]]] = [32, 64],
                 dropout: float = 0.0,
                 lr: float = 1e-3, epochs: int = 100,  temperature: float = 0.7,
                 lambda_r: float = 1.0, lambda_ent: float = 0.05, lambda_sp: float = 0.05,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.latent_dim = latent_dim
        self.view_latent_dim = view_latent_dim
        self.encoder_struct = encoder_struct
        self.decoder_struct = decoder_struct
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.temperature = temperature
        self.lambda_r = lambda_r
        self.lambda_ent = lambda_ent
        self.lambda_sp = lambda_sp
        self.epsilon = 1e-8

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.gating_net = None;
        self.compress_net = None;
        self.predict_net = None

    def _parse_struct_config(self, struct_config, num_views):
        if not struct_config: return [[] for _ in range(num_views)]
        if isinstance(struct_config[0], list):
            if len(struct_config) != num_views: raise ValueError("Config Error")
            return struct_config
        return [struct_config] * num_views

    def _init_networks(self, input_dims: Dict[str, int], device: str):
        self.modality_names = sorted(input_dims.keys())
        num_views = len(self.modality_names)
        enc_structs = self._parse_struct_config(self.encoder_struct, num_views)
        dec_structs = self._parse_struct_config(self.decoder_struct, num_views)
        total_view_latent = num_views * self.view_latent_dim

        for i, name in enumerate(self.modality_names):
            p_m = input_dims[name]
            self.encoders[name] = MLPBlock(p_m, self.view_latent_dim, enc_structs[i], self.dropout).to(device)
            self.decoders[name] = MLPBlock(self.latent_dim, p_m, dec_structs[i], self.dropout).to(device)

        self.gating_net = MLPBlock(total_view_latent, num_views, [64], self.dropout).to(device)
        self.compress_net = MLPBlock(total_view_latent, self.latent_dim, [64], self.dropout).to(device)
        self.predict_net = MLPBlock(self.latent_dim, 1, [32], self.dropout).to(device)

    def _marginal_ecdf_map(self, A: np.ndarray) -> np.ndarray:
        if A.ndim == 1: A = A[:, None]
        n_local, d_local = A.shape
        out = np.empty((n_local, d_local), dtype=float)
        for i in range(d_local):
            out[:, i] = rankdata(A[:, i], method="max") / n_local
        return out

    def _calculate_feature_scores(self, X_modality: np.ndarray, Z: np.ndarray) -> np.ndarray:
        n_local, p_local = X_modality.shape
        if np.var(Z) < 1e-9: return np.zeros(p_local)
        Zt = self._marginal_ecdf_map(Z)
        Xt = self._marginal_ecdf_map(X_modality)
        sc = np.zeros(p_local, dtype=float)
        for j in range(p_local):
            if np.var(Xt[:, j]) < 1e-9: continue
            try:
                sc[j] = dcor.distance_correlation(Xt[:, j], Zt)
            except:
                sc[j] = 0.0
        return sc

    def _perform_screening(self, X_dict, Z_np, n_samples):
        k_total = max(1, floor(n_samples / log(n_samples)) if n_samples > 1 else 1)
        total_dims = sum(x.shape[1] for x in X_dict.values())
        selected = {}
        all_scores = {k: self._calculate_feature_scores(v, Z_np) for k, v in X_dict.items()}
        for name, data in X_dict.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            idx = np.argsort(-all_scores[name])[:k_m]
            selected[name] = idx.tolist()
        return selected

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray,
            true_features: Optional[Dict[str, List[int]]] = None,
            verbose: bool = False) -> FitResult:
        """
        :param verbose: 是否打印 Debug 信息 (仅建议在 seed=0 时开启)
        """

        input_dims = {k: v.shape[1] for k, v in X.items()}
        n_samples = len(y)
        self._init_networks(input_dims, DEVICE)

        X_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in X.items()}
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)

        params = list(self.encoders.parameters()) + list(self.decoders.parameters()) + \
                 list(self.gating_net.parameters()) + list(self.compress_net.parameters()) + \
                 list(self.predict_net.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        criterion_mse = nn.MSELoss()

        # State Tracking
        best_recall_total = -1.0
        best_epoch_metrics = {}

        self.encoders.train();
        self.decoders.train();
        self.gating_net.train()
        self.compress_net.train();
        self.predict_net.train()

        if verbose:
            print(f"\n[{self.name}] Start Training (Total Epochs: {self.epochs})...")
            # 修正表头: 包含详细 Loss 构成
            print(
                f"{'Epoch':<6} | {'Tot_Loss':<10} | {'Pred_Loss':<10} | {'Rec_Loss':<10} | {'Ent_Loss':<10} | {'Sp_Loss':<10} | {'Rec_Tot':<8}")
            print("-" * 80)

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            # --- Forward ---
            rs = [self.encoders[name](X_t[name]) for name in self.modality_names]
            r_concat = torch.cat(rs, dim=1)
            gate_logits = self.gating_net(r_concat)
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)

            loss_ent = -torch.sum(alpha * torch.log(alpha + self.epsilon), dim=1).mean()
            loss_sp = torch.sum(alpha ** 2, dim=1).mean()

            weighted_rs = [rs[i] * alpha[:, i:i + 1] for i in range(len(rs))]
            z = self.compress_net(torch.cat(weighted_rs, dim=1))

            recon_loss = sum([criterion_mse(self.decoders[name](z), X_t[name]) for name in self.modality_names]) / len(
                self.modality_names)
            loss_s = criterion_mse(self.predict_net(z), y_t)

            total_loss = loss_s + self.lambda_r * recon_loss + self.lambda_ent * loss_ent + self.lambda_sp * loss_sp
            total_loss.backward()
            optimizer.step()

            # --- Debug / Tracking (Every 20 epochs or Last Epoch) ---
            if true_features and (epoch % 20 == 0 or epoch == self.epochs):
                # Eval Mode
                self.encoders.eval();
                self.compress_net.eval();
                self.gating_net.eval()
                with torch.no_grad():
                    rs_eval = [self.encoders[name](X_t[name]) for name in self.modality_names]
                    w_rs_eval = [rs_eval[i] * alpha[:, i:i + 1] for i in range(len(rs_eval))]
                    z_eval = self.compress_net(torch.cat(w_rs_eval, dim=1))
                    Z_np_eval = z_eval.cpu().numpy()

                # Screen
                sel_temp = self._perform_screening(X, Z_np_eval, n_samples)
                metrics = calculate_selection_metrics(true_features, sel_temp)
                metrics["epoch"] = epoch

                # Update Best (Logic remains: Track Best for Summary Report)
                if metrics["recall_total"] >= best_recall_total:
                    best_recall_total = metrics["recall_total"]
                    best_epoch_metrics = metrics.copy()

                # Print log (Logic Updated: Show detailed losses)
                if verbose:
                    print(
                        f"{epoch:<6} | {total_loss.item():<10.4f} | {loss_s.item():<10.4f} | {recon_loss.item():<10.4f} | {loss_ent.item():<10.4f} | {loss_sp.item():<10.4f} | {metrics['recall_total']:<8.4f}")

                # Restore Train Mode
                self.encoders.train();
                self.compress_net.train();
                self.gating_net.train()

        # ==========================
        # Final Inference
        # ==========================
        self.encoders.eval();
        self.compress_net.eval();
        self.gating_net.eval()
        with torch.no_grad():
            rs = [self.encoders[name](X_t[name]) for name in self.modality_names]
            gate_logits = self.gating_net(torch.cat(rs, dim=1))
            alpha = torch.softmax(gate_logits / self.temperature, dim=1)
            w_rs = [rs[i] * alpha[:, i:i + 1] for i in range(len(rs))]
            z_final = self.compress_net(torch.cat(w_rs, dim=1))
            Z_np = z_final.cpu().numpy()
            alpha_np = alpha.cpu().numpy()

        selected = self._perform_screening(X, Z_np, n_samples)

        return FitResult(
            selected_features=selected,
            model_state={
                "alpha": alpha_np,
                "z": Z_np,
                "best_epoch_metrics": best_epoch_metrics
            }
        )