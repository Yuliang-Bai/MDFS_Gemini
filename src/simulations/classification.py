import numpy as np
from typing import Tuple, Dict, List, Any
from src.methods.classification.proposed import MDFSClassifier
from src.methods.classification.baselines import SCFS, HLRFS, LSRFS
from src.utils.metrics import calculate_selection_metrics


def generate_ar_noise(n_samples: int, n_features: int, rho: float, rng) -> np.ndarray:
    """
    生成符合 Σ_ij = rho^|i-j| 的噪声矩阵 E (复用自 Regression)
    """
    if rho == 0:
        return rng.standard_normal((n_samples, n_features))
    noise = np.zeros((n_samples, n_features))
    noise[:, 0] = rng.standard_normal(n_samples)
    scale = np.sqrt(1 - rho ** 2)
    white = rng.standard_normal((n_samples, n_features))
    for j in range(1, n_features):
        noise[:, j] = rho * noise[:, j - 1] + scale * white[:, j]
    return noise


def generate_classification_data(n_samples: int = 200, n_features: List[int] = [300, 300],
                                 n_classes: int = 2,  # 默认为2，对应 01 Logit 分布
                                 noise_level: float = 1.0, gamma: float = 0.7, seed: int = 42):
    """
    基于 Logit 模型生成的二分类数据
    P(Y=1|X) = Sigmoid( Sum(X_v * beta_v) )
    """
    rng = np.random.default_rng(seed)
    n_views = len(n_features)

    # --- 1. 生成潜变量 (Latent Factors) - 与 Regression 保持一致 ---
    k0 = 5  # 共享维度
    ks = 2  # 模态特定维度
    H0 = rng.standard_normal((n_samples, k0))
    H_specs = [rng.standard_normal((n_samples, ks)) for _ in range(n_views)]

    # --- 2. 生成 Logit 响应变量 y ---
    # 生成线性部分 (Linear Response)
    beta0 = rng.uniform(0.5, 1.5, size=(k0, 1))
    linear_response = H0 @ beta0
    for h in H_specs:
        linear_response += h @ rng.uniform(0.5, 1.5, size=(ks, 1))

    # 转换为概率 (Sigmoid)
    logits = linear_response.flatten()
    # 归一化 logits 防止 sigmoid 饱和导致全0或全1
    logits = (logits - logits.mean()) / (logits.std() + 1e-8) * 2.0
    probs = 1.0 / (1.0 + np.exp(-logits))

    # 采样生成 0/1 标签
    y = rng.binomial(1, probs).astype(int)

    # --- 3. 生成特征矩阵 X (含 Signal + AR Noise) ---
    X_data = {}
    true_features = {}
    n_active_shared = 5
    n_active_spec = 5
    rho_val = 0.5

    for i, p in enumerate(n_features):
        name = f"view{i + 1}"

        # A. GT 索引
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)
        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        # B. 信号 X_sig
        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal((ks, n_active_spec))
        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[i] @ Ws)

        # C. 噪声 E
        E_m = generate_ar_noise(n_samples, p, rho_val, rng)

        # D. 叠加
        X_data[name] = X_signal + E_m * noise_level

    return X_data, y, true_features


def run_simulation_task_cls(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    X, y, true_feats = generate_classification_data(
        n_samples=config["n_samples"], n_features=config["dims"],
        n_classes=config.get("n_classes", 2),
        noise_level=config["noise"], gamma=config["gamma"], seed=seed
    )

    res = {"seed": seed}

    # --- 1. MDFS ---
    try:
        is_debug_seed = (seed == 0)
        model = MDFSClassifier(**config["mdfs_params"])
        fit_res = model.fit(X, y, true_features=true_feats, verbose=is_debug_seed)

        final_metrics = calculate_selection_metrics(true_feats, fit_res.selected_features)
        for k, v in final_metrics.items():
            res[f"MDFS_Final_{k}"] = v

        best_metrics = fit_res.model_state.get("best_epoch_metrics", {})
        if not best_metrics: best_metrics = final_metrics
        for k, v in best_metrics.items():
            if k == "epoch":
                res["MDFS_Best_Epoch"] = v
            else:
                res[f"MDFS_Best_{k}"] = v

    except Exception as e:
        print(f"MDFS Error: {e}")
        for k in ["recall_total", "precision_total"]:
            res[f"MDFS_Final_{k}"] = 0.0

    # --- 2. Baselines (SCFS, HLRFS, LSRFS) ---
    # 注意: 大多数基准方法对二分类同样适用 (Treat labels as regression target or class indicator)
    for name, Cls, params in [
        ("SCFS", SCFS, config["scfs_params"]),
        ("HLRFS", HLRFS, config["hlrfs_params"]),
        ("LSRFS", LSRFS, config["lsrfs_params"])
    ]:
        try:
            m = Cls(**params)
            f_res = m.fit(X, y)
            metrics = calculate_selection_metrics(true_feats, f_res.selected_features)
            for k, v in metrics.items():
                res[f"{name}_{k}"] = v
        except Exception as e:
            print(f"{name} Error: {e}")
            res[f"{name}_recall_total"] = 0.0
            res[f"{name}_precision_total"] = 0.0

    return res