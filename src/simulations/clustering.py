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


def generate_clustering_data(n_samples=200, n_features=[300, 300], n_clusters=3,
                             noise_level: float = 1.0, gamma: float = 0.7, seed: int = 42):
    """
    生成聚类模拟数据
    """
    rng = np.random.default_rng(seed)
    n_views = len(n_features)

    # ============ 1. 参数设定 ============
    k0 = 3  # 共享潜变量维度
    k1 = 2  # 模态1特异潜变量维度
    k2 = 2  # 模态2特异潜变量维度
    K = n_clusters  # 类别数

    # 每个模态的共享和特异信号特征数量
    n_active_shared = 5
    n_active_spec = 5

    # ============ 2. 生成类别标签（等概率）============
    y = rng.choice(K, size=n_samples, p=[1 / K] * K)

    # ============ 3. 设定类别中心 ============
    # 共享潜变量中心: μ_k = 3e_k
    mu_centers = np.zeros((K, k0))
    for k in range(K):
        mu_centers[k, k] = 3.0

    # 模态1特异潜变量中心: ν_k^(1) = e_k
    nu1_centers = np.zeros((K, k1))
    for k in range(K):
        if k < k1:  # k1=2，所以只处理k=0,1
            nu1_centers[k, k] = 1.0
        # k=2时保持为零向量

    # 模态2特异潜变量中心: ν_k^(2) = e_{k+1}
    nu2_centers = np.zeros((K, k2))
    for k in range(K):
        if k + 1 < k2:  # k2=2，所以k+1=1
            nu2_centers[k, k + 1] = 1.0
        # k=1时k+1=2超出范围，保持为零向量
        # k=2时k+1=3超出范围，保持为零向量

    # ============ 4. 生成潜变量（给定类别下）============
    H0 = np.zeros((n_samples, k0))
    H1 = np.zeros((n_samples, k1))
    H2 = np.zeros((n_samples, k2))

    for i in range(n_samples):
        k = y[i]
        H0[i] = rng.multivariate_normal(mu_centers[k], np.eye(k0))
        H1[i] = rng.multivariate_normal(nu1_centers[k], np.eye(k1))
        H2[i] = rng.multivariate_normal(nu2_centers[k], np.eye(k2))

    H_specs = [H1, H2]  # 用于后续循环

    # ============ 5. 为每个模态生成数据 ============
    X_data = {}
    true_features = {}

    for i, p in enumerate(n_features):
        name = f"view{i + 1}"

        # A. 确定真实特征索引
        all_idx = np.arange(p)
        idx_shared = rng.choice(all_idx, n_active_shared, replace=False)
        rem_idx = np.setdiff1d(all_idx, idx_shared)
        idx_spec = rng.choice(rem_idx, n_active_spec, replace=False)
        true_features[name] = np.union1d(idx_shared, idx_spec).tolist()

        # B. 生成载荷矩阵
        # 共享载荷: w_{0m,j} ~ N(0, I_{k0}), ∀j ∈ S_sh
        W0 = rng.standard_normal((k0, n_active_shared))
        Ws = rng.standard_normal(([k1,k2][i], n_active_spec))
        X_signal = np.zeros((n_samples, p))
        X_signal[:, idx_shared] += gamma * (H0 @ W0)
        X_signal[:, idx_spec] += (1 - gamma) * (H_specs[i] @ Ws)

        # C. 生成AR(1)噪声 E(m) ~ N(0, Σ_m), Σ_ij = ρ^|i-j|
        rho = 0.5
        E_m = generate_noise(n_samples, p, rho, rng)

        # E. 最终数据: X(m) = X_sig(m) + E(m)
        X_data[name] = X_signal + E_m * noise_level

    return X_data, y, true_features


def run_simulation_task_clu(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行单次聚类模拟任务
    """
    X, y, true_feats = generate_clustering_data(
        n_samples=config["n_samples"],
        n_features=config["dims"],
        n_clusters=config.get("n_clusters", 3),
        seed=seed
    )

    res = {"seed": seed}

    # --- 1. MDFS (Proposed) ---
    try:
        # [关键修复] 只有在 seed=0 时才开启 verbose，避免并行时输出混乱
        is_debug_seed = (seed == 0)

        model = MDFSClustering(**config["mdfs_params"])

        # 传入 verbose=is_debug_seed
        fit_res = model.fit(X, true_features=true_feats, verbose=is_debug_seed)

        # 1.1 获取 MDFS_Final 指标
        metrics_final = calculate_selection_metrics(true_feats, fit_res.selected_features)
        for k, v in metrics_final.items():
            res[f"MDFS_Final_{k}"] = v

        # 1.2 获取 MDFS_Best 指标 (从 model_state 中提取)
        best_metrics = fit_res.model_state.get("best_epoch_metrics", {})
        if not best_metrics:
            best_metrics = metrics_final  # Fallback

        for k, v in best_metrics.items():
            # 过滤掉不需要的 epoch 字段，或者保留它
            if k == "epoch":
                res["MDFS_Best_Epoch"] = v
            else:
                res[f"MDFS_Best_{k}"] = v

    except Exception as e:
        # 如果你想看具体的报错，可以取消下面这行的注释
        # print(f"Seed {seed} MDFS Error: {e}")
        res["MDFS_Final_recall_total"] = 0.0
        res["MDFS_Final_precision_total"] = 0.0
        res["MDFS_Best_recall_total"] = 0.0
        res["MDFS_Best_precision_total"] = 0.0

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