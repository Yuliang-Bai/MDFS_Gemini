import numpy as np
from typing import Dict, List, Any
from src.methods.clustering.proposed import MDFSClustering
from src.methods.clustering.baselines import MCFL, MRAG
from src.utils.metrics import calculate_selection_metrics

# 尝试导入 NSGL
try:
    from src.methods.clustering.baselines import NSGL
except ImportError:
    NSGL = None


def generate_noise(n_samples: int, n_features: int, rho: float, rng) -> np.ndarray:
    indices = np.arange(n_features)
    cov_matrix = rho ** np.abs(indices[:, None] - indices[None, :])
    try:
        L = np.linalg.cholesky(cov_matrix)
    except:
        L = np.linalg.cholesky(cov_matrix + np.eye(n_features) * 1e-5)
    Z = rng.standard_normal((n_samples, n_features))
    E = Z @ L.T
    return E


def generate_clustering_data(n_samples=200, n_features=[300, 300], n_clusters=3,
                             noise_level: float = 1.0, gamma: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_views = len(n_features)

    k0 = 3
    k1 = 2
    k2 = 2
    K = n_clusters
    n_active_shared = 5
    n_active_spec = 5

    y = rng.choice(K, size=n_samples, p=[1 / K] * K)

    mu_centers = np.zeros((K, k0))
    for k in range(K): mu_centers[k, k] = 3.0

    nu1_centers = np.zeros((K, k1))
    for k in range(K):
        if k < k1: nu1_centers[k, k] = 1.0

    nu2_centers = np.zeros((K, k2))
    for k in range(K):
        if k + 1 < k2: nu2_centers[k, k + 1] = 1.0

    H0 = np.zeros((n_samples, k0))
    H1 = np.zeros((n_samples, k1))
    H2 = np.zeros((n_samples, k2))

    for i in range(n_samples):
        k = y[i]
        H0[i] = rng.multivariate_normal(mu_centers[k], np.eye(k0))
        H1[i] = rng.multivariate_normal(nu1_centers[k], np.eye(k1))
        H2[i] = rng.multivariate_normal(nu2_centers[k], np.eye(k2))

    H_specs = [H1, H2]

    X_data = {}
    true_features = {}

    for i, p in enumerate(n_features):
        name = f"view{i + 1}"
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)
        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        dim_spec = k1 if i == 0 else k2
        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal((dim_spec, n_active_spec))

        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[i] @ Ws)

        E_m = generate_noise(n_samples, p, 0.5, rng)
        X_data[name] = X_signal + E_m * noise_level

    return X_data, y, true_features


def run_simulation_task_clu(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    X, y, true_feats = generate_clustering_data(
        n_samples=config["n_samples"],
        n_features=config["dims"],
        n_clusters=config.get("n_clusters", 3),
        noise_level=config.get("noise_level", 1.0),
        seed=seed
    )

    res = {"seed": seed}

    # 只有 Seed 0 开启 Debug 模式
    is_debug_seed = (seed == 0)

    # --- 1. MDFS (Proposed) ---
    try:
        model = MDFSClustering(**config["mdfs_params"])
        fit_res = model.fit(X, true_features=true_feats, verbose=is_debug_seed)

        metrics_final = calculate_selection_metrics(true_feats, fit_res.selected_features)
        for k, v in metrics_final.items():
            res[f"MDFS_Final_{k}"] = v

        best_metrics = fit_res.model_state.get("best_epoch_metrics", {})
        if not best_metrics: best_metrics = metrics_final
        for k, v in best_metrics.items():
            if k == "epoch":
                res["MDFS_Best_Epoch"] = v
            else:
                res[f"MDFS_Best_{k}"] = v

    except Exception as e:
        if is_debug_seed: print(f"[MDFS Failed] Seed {seed}: {e}")
        for prefix in ["MDFS_Final", "MDFS_Best"]:
            res[f"{prefix}_recall_total"] = 0.0
            res[f"{prefix}_precision_total"] = 0.0
            res[f"{prefix}_f1_total"] = 0.0
            for view_name in X.keys():
                res[f"{prefix}_{view_name}_recall"] = 0.0
                res[f"{prefix}_{view_name}_precision"] = 0.0
                res[f"{prefix}_{view_name}_f1"] = 0.0

    # --- 2. Baselines ---
    baselines = [
        ("MCFL", MCFL, config.get("mcfl_params", {})),
        ("MRAG", MRAG, config.get("mrag_params", {}))
    ]
    if NSGL is not None:
        baselines.append(("NSGL", NSGL, config.get("nsgl_params", {})))

    for name, Cls, params in baselines:
        try:
            # 传递 verbose 参数来监控 seed=0 的运行
            m = Cls(**params)

            # 关键修改：传入 verbose 参数
            f_res = m.fit(X, verbose=is_debug_seed)

            metrics = calculate_selection_metrics(true_feats, f_res.selected_features)
            for k, v in metrics.items():
                res[f"{name}_{k}"] = v
        except Exception as e:
            print(f"[{name} Failed] Seed {seed}: {e}")
            res[f"{name}_recall_total"] = 0.0
            res[f"{name}_precision_total"] = 0.0
            res[f"{name}_f1_total"] = 0.0
            for view_name in X.keys():
                res[f"{name}_{view_name}_recall"] = 0.0
                res[f"{name}_{view_name}_precision"] = 0.0
                res[f"{name}_{view_name}_f1"] = 0.0

    return res