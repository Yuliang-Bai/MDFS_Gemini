import numpy as np
from typing import Tuple, Dict, List, Any
from src.methods.clustering.proposed import MDFSClustering
from src.methods.clustering.baselines import MCFL, MRAG, NSGL
from src.utils.metrics import calculate_selection_metrics


def generate_ar_noise(n_samples: int, n_features: int, rho: float, rng) -> np.ndarray:
    if rho == 0:
        return rng.standard_normal((n_samples, n_features))

    noise = np.zeros((n_samples, n_features))
    noise[:, 0] = rng.standard_normal(n_samples)
    scale = np.sqrt(1 - rho ** 2)
    white = rng.standard_normal((n_samples, n_features))

    for j in range(1, n_features):
        noise[:, j] = rho * noise[:, j - 1] + scale * white[:, j]

    return noise


def generate_clustering_data(n_samples: int = 200, n_features: List[int] = [300, 300],
                             noise_level: float = 1.0, gamma: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_views = len(n_features)
    n_clusters = 3

    k0 = 3
    ks = [4, 4]

    mus = np.zeros((n_clusters, k0))
    for k in range(n_clusters):
        mus[k, k] = 3.0

    nus = []
    nu1 = np.zeros((n_clusters, ks[0]))
    for k in range(n_clusters):
        nu1[k, k] = 1.0
    nus.append(nu1)

    nu2 = np.zeros((n_clusters, ks[1]))
    for k in range(n_clusters):
        if k + 1 < ks[1]:
            nu2[k, k + 1] = 1.0
    nus.append(nu2)

    y = rng.integers(0, n_clusters, size=n_samples)

    H0 = np.zeros((n_samples, k0))
    H_specs = [np.zeros((n_samples, ks[i])) for i in range(n_views)]

    for i in range(n_samples):
        c_i = y[i]
        H0[i] = rng.multivariate_normal(mus[c_i], np.eye(k0))
        for m in range(n_views):
            H_specs[m][i] = rng.multivariate_normal(nus[m][c_i], np.eye(ks[m]))

    X_data = {}
    true_features = {}
    n_active_shared = 5
    n_active_spec = 5
    rho_val = 0.5

    for m, p in enumerate(n_features):
        name = f"view{m + 1}"
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)
        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal((ks[m], n_active_spec))

        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[m] @ Ws)

        E_m = generate_ar_noise(n_samples, p, rho_val, rng)
        X_data[name] = X_signal + E_m * noise_level

    return X_data, y, true_features


def run_simulation_task(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    X, y, true_feats = generate_clustering_data(
        n_samples=config["n_samples"], n_features=config["dims"],
        noise_level=config["noise"], gamma=config["gamma"], seed=seed
    )

    res = {"seed": seed}

    # --- 1. MDFS (Proposed) ---
    try:
        is_debug_seed = (seed == 0)
        model = MDFSClustering(**config["mdfs_params"])
        fit_res = model.fit(X, y=None, true_features=true_feats, verbose=is_debug_seed)

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
            res[f"MDFS_Best_{k}"] = 0.0

    # --- 2. Baselines (Renamed) ---
    for name, Cls, params in [
        ("MCFL", MCFL, config.get("mcfl_params", {"n_clusters": 3})),
        ("MRAG", MRAG, config.get("mrag_params", {"n_clusters": 3})),
        ("NSGL", NSGL, config.get("nsgl_params", {"n_clusters": 3}))
    ]:
        try:
            m = Cls(**params)
            f_res = m.fit(X)
            metrics = calculate_selection_metrics(true_feats, f_res.selected_features)
            for k, v in metrics.items():
                res[f"{name}_{k}"] = v
        except Exception as e:
            print(f"{name} Error: {e}")
            res[f"{name}_recall_total"] = 0.0
            res[f"{name}_precision_total"] = 0.0

    return res