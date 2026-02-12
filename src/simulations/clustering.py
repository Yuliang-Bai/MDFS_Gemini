import numpy as np
from typing import Dict, List, Any
from src.methods.clustering.proposed import MDFSClustering
from src.methods.clustering.baselines import MCFL, MRAG
from src.utils.metrics import calculate_selection_metrics

# 尝试导入 NSGL (防止未实现导致报错)
try:
    from src.methods.clustering.baselines import NSGL
except ImportError:
    NSGL = None


def generate_clustering_data(n_samples=200, dims=[300, 300], n_clusters=3, seed=42):
    """
    生成聚类模拟数据
    """
    rng = np.random.default_rng(seed)
    X = {f"view{i + 1}": rng.standard_normal((n_samples, p)) for i, p in enumerate(dims)}
    y = rng.integers(0, n_clusters, size=n_samples)

    true_features = {}
    # 注入聚类结构
    for i, (name, data) in enumerate(X.items()):
        true_idx = [0, 1, 2, 3, 4]
        true_features[name] = true_idx
        for cls in range(n_clusters):
            mask = (y == cls)
            if np.sum(mask) > 0:
                shift = rng.uniform(-1.5, 1.5, size=len(true_idx))
                data[np.ix_(mask, true_idx)] += shift

    return X, y, true_features


def run_simulation_task_clu(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行单次聚类模拟任务
    """
    X, y, true_feats = generate_clustering_data(
        n_samples=config["n_samples"],
        dims=config["dims"],
        n_clusters=config.get("n_clusters", 3),
        seed=seed
    )

    res = {"seed": seed}

    # --- 1. MDFS (Proposed) ---
    try:
        model = MDFSClustering(**config["mdfs_params"])
        fit_res = model.fit(X)
        metrics = calculate_selection_metrics(true_feats, fit_res.selected_features)

        # [关键修改]：统一命名为 MDFS_Final_xxx，供脚本动态识别
        for k, v in metrics.items():
            res[f"MDFS_Final_{k}"] = v

    except Exception as e:
        # print(f"MDFS Error: {e}")
        res["MDFS_Final_recall_total"] = 0.0
        res["MDFS_Final_precision_total"] = 0.0

    # --- 2. Baselines ---
    baselines = [
        ("MCFL", MCFL, config.get("mcfl_params", {})),
        ("MRAG", MRAG, config.get("mrag_params", {}))
    ]
    if NSGL is not None:
        baselines.append(("NSGL", NSGL, config.get("nsgl_params", {})))

    for name, Cls, params in baselines:
        try:
            m = Cls(**params)
            f_res = m.fit(X)
            metrics = calculate_selection_metrics(true_feats, f_res.selected_features)
            for k, v in metrics.items():
                res[f"{name}_{k}"] = v
        except Exception as e:
            res[f"{name}_recall_total"] = 0.0
            res[f"{name}_precision_total"] = 0.0

    return res