import sys
import os
import multiprocessing
import pandas as pd
from tqdm import tqdm
from functools import partial

# ==========================================================
# 1. 路径修复
# ==========================================================
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'scripts':
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
    else:
        project_root = current_dir

if project_root not in sys.path:
    sys.path.append(project_root)

from src.methods.regression.proposed import MDFSRegressor
from src.methods.regression.baselines import AdaCoop, MSGLasso, SLRFS
from src.simulations.regression import run_simulation_task

# ==========================
# 2. 实验配置 (Experiment Config)
# ==========================
CONFIG = {
    # --- 模拟环境参数 ---
    "n_repeats": 20,
    "n_cores": 4,
    "n_samples": 200,
    "dims": [300, 300],  # [Modality_1_Dim, Modality_2_Dim]
    "noise": 0.5,
    "gamma": 0.7,

    # --- MDFS 超参数 (MDFS Hyperparameters) ---
    "mdfs_params": {
        "latent_dim": 5,  # Joint Latent Z Dimension
        "view_latent_dim": 10,  # View Latent R Dimension

        # 【关键修改】网络拓扑结构 (Network Topology)
        # 使用双重列表，分别对应 [Modality_1, Modality_2]

        # 编码器结构 (Encoder Structure):
        # View 1: Input -> Hidden_Layer_1(128) -> Hidden_Layer_2(64) -> R
        # View 2: Input -> Hidden_Layer_1(128) -> Hidden_Layer_2(64) -> R
        "encoder_struct": [
            [128, 64],  # For View 1
            [128, 64] # For View 2
        ],

        # 解码器结构 (Decoder Structure):
        # 同样分别指定，或使用单层列表广播
        # Joint Z -> Hidden_Layer_1(64) -> Output
        "decoder_struct": [
            [64, 128],  # For View 1 (Symmetric to encoder)
            [64, 128]  # For View 2
        ],

        "temperature": 0.5,
        "epochs": 200,
        "lr": 1e-3,
        "lambda_sp": 0.1
    },

    # --- 基准方法参数 ---
    "lasso_params": {},
    "slrfs_params": {}
}

if __name__ == "__main__":
    print(f"Starting Regression Simulation ({CONFIG['n_repeats']} repeats)...")
    print(f"MDFS Topology:")
    print(f"  - Encoder Structures: {CONFIG['mdfs_params']['encoder_struct']}")
    print(f"  - Decoder Structures: {CONFIG['mdfs_params']['decoder_struct']}")

    task = partial(run_simulation_task, config=CONFIG)

    if CONFIG["n_cores"] > 1:
        with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
            results = list(tqdm(pool.imap(task, range(CONFIG["n_repeats"])), total=CONFIG["n_repeats"]))
    else:
        results = [task(seed) for seed in range(CONFIG["n_repeats"])]

    df = pd.DataFrame(results)

    print("\n=== Simulation Results (Average Recall) ===")
    print(df[["MDFS", "AdaCoop", "MSGLasso", "SLRFS"]].mean())
    print("\n=== Standard Deviation ===")
    print(df[["MDFS", "AdaCoop", "MSGLasso", "SLRFS"]].std())

    save_dir = os.path.join(project_root, "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sim_1_reg.csv")
    df.to_csv(save_path, index=False)
    print(f"\nDetails saved to {save_path}")