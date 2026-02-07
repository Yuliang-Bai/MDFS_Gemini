import numpy as np
from typing import Tuple, Dict, List, Any
# 使用绝对路径导入，避免相对路径在不同层级调用时出错
from src.methods.regression.proposed import MDFSRegressor
from src.methods.regression.baselines import AdaCoop, MSGLasso, SLRFS
from src.utils.metrics import calculate_recall


def generate_ar_noise(n_samples: int, n_features: int, rho: float, rng) -> np.ndarray:
    """生成具有 AR(1) 相关结构的噪声矩阵"""
    if rho == 0:
        return rng.standard_normal((n_samples, n_features))

    noise = np.zeros((n_samples, n_features))
    noise[:, 0] = rng.standard_normal(n_samples)
    scale = np.sqrt(1 - rho ** 2)
    white = rng.standard_normal((n_samples, n_features))

    for j in range(1, n_features):
        noise[:, j] = rho * noise[:, j - 1] + scale * white[:, j]

    return noise


def generate_regression_data(n_samples: int = 200,
                             n_features: List[int] = [300, 500],
                             noise_level: float = 1.0,
                             gamma: float = 0.7,
                             rho: float = 0.5,
                             seed: int = 42) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, List[int]]]:
    rng = np.random.default_rng(seed)
    n_views = len(n_features)

    # 1. 生成潜因子
    k0 = 3
    ks = 2

    H0 = rng.standard_normal((n_samples, k0))
    H_specs = [rng.standard_normal((n_samples, ks)) for _ in range(n_views)]

    # 2. 生成响应变量
    beta0 = rng.uniform(1, 2, size=(k0, 1))
    y_signal = H0 @ beta0
    for h in H_specs:
        beta_s = rng.uniform(1, 2, size=(ks, 1))
        y_signal += h @ beta_s

    y = y_signal.flatten() + rng.standard_normal(n_samples)

    # 3. 生成模态数据
    X_data = {}
    true_features = {}

    n_active_shared = 5
    n_active_spec = 5

    for i, p in enumerate(n_features):
        name = f"view{i + 1}"
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)

        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal((ks, n_active_spec))

        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[i] @ Ws)

        E = generate_ar_noise(n_samples, p, rho, rng)
        X_data[name] = X_signal + E

    return X_data, y, true_features


def run_simulation_task(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    多进程 Worker 函数。
    必须定义在模块顶层，以便子进程可以 import。
    """
    # 1. 生成数据
    X, y, true_feats = generate_regression_data(
        n_samples=config["n_samples"],
        n_features=config["dims"],
        noise_level=config["noise"],
        gamma=config["gamma"],
        seed=seed
    )

    res = {"seed": seed}

    # 2. 运行 MDFS
    try:
        model = MDFSRegressor(**config["mdfs_params"])
        fit_res = model.fit(X, y)
        res["MDFS"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except Exception as e:
        res["MDFS"] = 0.0

    # 3. 运行 AdaCoop
    try:
        model = AdaCoop(**config["lasso_params"])
        fit_res = model.fit(X, y)
        res["AdaCoop"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["AdaCoop"] = 0.0

    # 4. 运行 MSGLasso
    try:
        model = MSGLasso(**config["lasso_params"])
        fit_res = model.fit(X, y)
        res["MSGLasso"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["MSGLasso"] = 0.0

    # 5. 运行 SLRFS
    try:
        model = SLRFS(**config["slrfs_params"])
        fit_res = model.fit(X, y)
        res["SLRFS"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["SLRFS"] = 0.0

    return res