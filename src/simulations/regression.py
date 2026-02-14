import numpy as np
from typing import Tuple, Dict, List, Any
from src.methods.regression.proposed import MDFSRegressor
from src.methods.regression.baselines import AdaCoop, MSGLasso, SLRFS
from src.utils.metrics import calculate_selection_metrics


def generate_noise(n_samples: int, n_features: int, rho: float, rng) -> np.ndarray:
    """
    生成符合 Σ_ij = rho^|i-j| 的噪声矩阵 E
    """
    indices = np.arange(n_features)
    cov_matrix = rho ** np.abs(indices[:, None] - indices[None, :])

    # 2. Cholesky分解
    L = np.linalg.cholesky(cov_matrix)

    # 3. 生成并变换
    Z = rng.standard_normal((n_samples, n_features))
    E = Z @ L.T

    return E


def generate_regression_data(n_samples: int = 200, n_features: List[int] = [300, 500],
                             noise_level: float = 1.0, gamma: float = 0.7, seed: int = 42):
    """
    符合数学公式 X(m) = X_sig(m) + E(m) 的数据生成函数
    """
    rng = np.random.default_rng(seed)
    n_views = len(n_features)

    # --- 1. 生成潜变量 (Latent Factors) ---
    k0 = 3  # 共享维度
    ks = 2  # 模态特定维度
    H0 = rng.standard_normal((n_samples, k0))
    H_specs = [rng.standard_normal((n_samples, ks)) for _ in range(n_views)]

    # --- 2. 生成响应变量 y (Signal) ---
    beta0 = rng.uniform(1, 2, size=(k0, 1))
    y_signal = H0 @ beta0
    for h in H_specs:
        y_signal += h @ rng.uniform(1, 2, size=(ks, 1))
    y = y_signal.flatten() + rng.standard_normal(n_samples)  # y 依然带有一点独立噪声

    X_data = {}
    true_features = {}
    n_active_shared = 5  # 每模态共享特征数
    n_active_spec = 5  # 每模态特定特征数

    # 设定自相关系数 rho = 0.5
    rho_val = 0.5

    # --- 3. 为每个模态生成信号与自相关噪声 ---
    for i, p in enumerate(n_features):
        name = f"view{i + 1}"

        # A. 确定真实特征索引 (Ground Truth)
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)
        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        # B. 生成模态信号 X_sig
        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal((ks, n_active_spec))
        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[i] @ Ws)

        # C. 生成自相关噪声 E(m)
        # 这里直接调用你确认过的 AR(1) 噪声生成函数
        # 它保证了每一行噪声向量 e ~ N(0, Σ)，且 Σ_ij = 0.5^|i-j|
        E_m = generate_noise(n_samples, p, rho_val, rng)

        # D. 最终叠加: X(m) = X_sig(m) + E(m)
        # noise_level 用于整体控制 E 的强弱（通常设为 1.0）
        X_data[name] = X_signal + E_m * noise_level

    return X_data, y, true_features


def run_simulation_task(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    X, y, true_feats = generate_regression_data(
        n_samples=config["n_samples"], n_features=config["dims"],
        noise_level=config["noise"], gamma=config["gamma"], seed=seed
    )

    res = {"seed": seed}

    # --- 1. MDFS (Enable verbose only for seed 0) ---
    try:
        is_debug_seed = (seed == 0)
        model = MDFSRegressor(**config["mdfs_params"])
        fit_res = model.fit(X, y, true_features=true_feats, verbose=is_debug_seed)

        # A. Final Results (End of training)
        final_metrics = calculate_selection_metrics(true_feats, fit_res.selected_features)
        for k, v in final_metrics.items():
            res[f"MDFS_Final_{k}"] = v

        # B. Best Epoch Results
        best_metrics = fit_res.model_state.get("best_epoch_metrics", {})
        if not best_metrics: best_metrics = final_metrics  # Fallback

        for k, v in best_metrics.items():
            if k == "epoch":
                res["MDFS_Best_Epoch"] = v
            else:
                res[f"MDFS_Best_{k}"] = v

    except Exception as e:
        print(f"MDFS Error: {e}")
        # Fill zeros for safety
        for k in ["recall_total", "precision_total"]:
            res[f"MDFS_Final_{k}"] = 0.0
            res[f"MDFS_Best_{k}"] = 0.0

    # --- 2. Baselines ---
    for name, Cls, params in [
        ("AdaCoop", AdaCoop, config["ada_params"]),
        ("MSGLasso", MSGLasso, config["msg_params"]),
        ("SLRFS", SLRFS, config["slrfs_params"])
    ]:
        try:
            m = Cls(**params)
            f_res = m.fit(X, y)
            metrics = calculate_selection_metrics(true_feats, f_res.selected_features)
            for k, v in metrics.items():
                res[f"{name}_{k}"] = v
        except:
            res[f"{name}_recall_total"] = 0.0
            res[f"{name}_precision_total"] = 0.0

    return res