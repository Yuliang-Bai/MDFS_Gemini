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

# 从模块中导入 Worker 函数
from src.simulations.regression import run_simulation_task

# ==========================
# 2. 实验配置
# ==========================
CONFIG = {
    "n_repeats": 20,
    "n_cores": 4,
    "n_samples": 200,
    "dims": [300, 300],
    "noise": 1.0,
    "gamma": 0.7,

    "mdfs_params": {"latent_dim": 10, "temperature": 0.5},
    "lasso_params": {},
    "slrfs_params": {}
}

if __name__ == "__main__":
    print(f"Starting Regression Simulation ({CONFIG['n_repeats']} repeats)...")

    # 使用 partial 固定 config 参数，因为 map 只能传一个变动参数 (seed)
    task = partial(run_simulation_task, config=CONFIG)

    if CONFIG["n_cores"] > 1:
        with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
            # 这里的 imap 会自动去 pickle 'src.simulations.regression.run_simulation_task'
            # 而不是 '__main__.run_single_trial'，从而解决 AttributeError
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