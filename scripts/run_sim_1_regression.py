import sys
import os

# 确保可以导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing
import pandas as pd
from tqdm import tqdm
from src.methods.regression.proposed import MDFSRegressor
from src.methods.regression.baselines import AdaCoop, MSGLasso, SLRFS
from src.simulations.regression import generate_regression_data
from src.utils.metrics import calculate_recall

# ==========================
# 实验配置 (超参数)
# ==========================
CONFIG = {
    "n_repeats": 20,  # 调试时设为 20，正式跑设为 500
    "n_cores": 4,  # 并行核数
    "n_samples": 200,
    "dims": [300, 300],  # 两个模态，维度各300
    "noise": 1.0,
    "gamma": 0.7,  # 共享信号比例

    # 方法参数
    "mdfs_params": {"latent_dim": 10, "temperature": 0.5},
    "lasso_params": {},
    "slrfs_params": {}
}


def run_single_trial(seed):
    # 1. 生成数据
    X, y, true_feats = generate_regression_data(
        n_samples=CONFIG["n_samples"],
        n_features=CONFIG["dims"],
        noise_level=CONFIG["noise"],
        gamma=CONFIG["gamma"],
        seed=seed
    )

    res = {}
    res["seed"] = seed

    # 2. 运行 MDFS
    try:
        model = MDFSRegressor(**CONFIG["mdfs_params"])
        fit_res = model.fit(X, y)
        res["MDFS"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except Exception as e:
        res["MDFS"] = 0.0
        print(f"MDFS Error: {e}")

    # 3. 运行 AdaCoop
    try:
        model = AdaCoop(**CONFIG["lasso_params"])
        fit_res = model.fit(X, y)
        res["AdaCoop"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["AdaCoop"] = 0.0

    # 4. 运行 MSGLasso
    try:
        model = MSGLasso(**CONFIG["lasso_params"])
        fit_res = model.fit(X, y)
        res["MSGLasso"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["MSGLasso"] = 0.0

    # 5. 运行 SLRFS
    try:
        model = SLRFS(**CONFIG["slrfs_params"])
        fit_res = model.fit(X, y)
        res["SLRFS"] = calculate_recall(true_feats, fit_res.selected_features)["recall_total"]
    except:
        res["SLRFS"] = 0.0

    return res


if __name__ == "__main__":
    print(f"Starting Regression Simulation ({CONFIG['n_repeats']} repeats)...")

    with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(pool.imap(run_single_trial, range(CONFIG["n_repeats"])), total=CONFIG["n_repeats"]))

    df = pd.DataFrame(results)

    print("\n=== Simulation Results (Average Recall) ===")
    print(df[["MDFS", "AdaCoop", "MSGLasso", "SLRFS"]].mean())
    print("\n=== Standard Deviation ===")
    print(df[["MDFS", "AdaCoop", "MSGLasso", "SLRFS"]].std())

    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/sim_1_reg.csv", index=False)
    print("\nDetails saved to results/sim_1_reg.csv")