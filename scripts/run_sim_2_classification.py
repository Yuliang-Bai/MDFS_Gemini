import sys
import os
import multiprocessing
import pandas as pd
from tqdm import tqdm
from functools import partial

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

from src.simulations.classification import run_simulation_task_cls

# 配置: 01 分布 (n_classes=2)
CONFIG = {
    "n_repeats": 20, "n_cores": 4,
    "n_samples": 200, "dims": [300, 300],
    "n_classes": 2, "noise": 1.0, "gamma": 0.7,
    "mdfs_params": {
        "latent_dim": 8, "view_latent_dim": 16,
        "encoder_struct": [[128], [64]],
        "decoder_struct": [[64], [128]],
        "temperature": 0.5, "epochs": 150, "lr": 5e-3,
        "lambda_sp": 0.1, "lambda_ent": 0.05
    },
    "scfs_params": {"alpha": 0.5, "rho": 1.0, "max_iter": 50},
    "hlrfs_params": {"beta": 1.0, "gamma": 0.5, "n_neighbors": 5},
    "lsrfs_params": {"lambda1": 0.1, "lambda2": 0.1, "max_iter": 30}
}

if __name__ == "__main__":
    print(f"Starting Classification Sim (Repeats: {CONFIG['n_repeats']}, Classes: {CONFIG['n_classes']})...")
    print("-" * 60)

    task = partial(run_simulation_task_cls, config=CONFIG)

    if CONFIG["n_cores"] > 1:
        with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
            results = list(tqdm(pool.imap(task, range(CONFIG['n_repeats'])), total=CONFIG['n_repeats']))
    else:
        results = [task(s) for s in range(CONFIG['n_repeats'])]

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(" >>> Final Classification Report (Mean ± SD)")
    print("=" * 80)

    methods = ["MDFS_Best", "MDFS_Final", "SCFS", "HLRFS", "LSRFS"]
    metrics_to_show = ["recall_total", "precision_total"]

    summary_rows = []
    for method in methods:
        row_data = {"Method": method}
        for metric in metrics_to_show:
            col_name = f"{method}_{metric}"
            if col_name in df.columns:
                mean_val = df[col_name].mean()
                sd_val = df[col_name].std()
                row_data[f"{metric}_Mean"] = mean_val
                row_data[f"{metric}_SD"] = sd_val
            else:
                row_data[f"{metric}_Mean"] = 0.0
                row_data[f"{metric}_SD"] = 0.0
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    if "MDFS_Best_Epoch" in df.columns:
        print(f"\n[Info] MDFS Avg Best Epoch: {df['MDFS_Best_Epoch'].mean():.2f}")

    print(summary_df.set_index("Method"))

    save_path = os.path.join(project_root, "results", "sim_2_cls_summary.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"\nSaved: {save_path}")