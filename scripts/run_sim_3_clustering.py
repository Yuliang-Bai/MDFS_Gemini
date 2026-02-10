import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulations.clustering import run_simulation_task

# ==========================================
# 实验配置 (Configuration)
# ==========================================
CONFIG = {
    "task_name": "Clustering_Simulation",
    "n_samples": 200,
    "dims": [300, 300],
    "noise": 1.0,
    "gamma": 0.7,

    # --- MDFS (Proposed) 参数 ---
    "mdfs_params": {
        "latent_dim": 16,
        "view_latent_dim": 16,
        "encoder_struct": [64, 32],
        "decoder_struct": [32, 64],
        "epochs": 100,
        "lr": 1e-3,
        "batch_size": 32,
        "lambda_r": 1.0,
        "lambda_ent": 0.05,
        "lambda_sp": 0.05,
        "temperature": 0.7
    },

    # --- Baselines 参数 ---
    # 1. MCFL (Wang et al., 2013)
    "mcfl_params": {
        "n_clusters": 3,
        "gamma": 0.1,
        "max_iter": 20
    },

    # 2. MRAG (Jing et al., 2021)
    "mrag_params": {
        "n_clusters": 3,
        "k_neighbors": 5
    },

    # 3. NSGL (Bai et al., 2020)
    "nsgl_params": {
        "n_clusters": 3,
        "alpha": 0.1
    }
}

SEEDS = range(10)
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    print(f"Start Simulation: {CONFIG['task_name']}")
    print(f"Config: Dims={CONFIG['dims']}, Samples={CONFIG['n_samples']}")

    for seed in tqdm(SEEDS, desc="Running Seeds"):
        res = run_simulation_task(seed, CONFIG)
        all_results.append(res)

    df = pd.DataFrame(all_results)

    csv_path = os.path.join(RESULTS_DIR, "sim_3_clustering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n=== Summary Metrics (Recall & Precision) ===")
    metric_cols = [c for c in df.columns if "recall" in c or "precision" in c]
    means = df[metric_cols].mean()
    stds = df[metric_cols].std()

    for col in sorted(metric_cols):
        print(f"{col:<30}: {means[col]:.4f} ± {stds[col]:.4f}")


if __name__ == "__main__":
    main()