import sys
import os
import multiprocessing
import pandas as pd
from tqdm import tqdm
from functools import partial

# --- Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

from src.simulations.clustering import run_simulation_task_clu

# --- Configuration ---
CONFIG = {
    "task_name": "Clustering_Simulation",
    "n_repeats": 20,
    "n_cores": 4,
    "n_samples": 200,
    "dims": [300, 300],
    "n_clusters": 3,
    "mdfs_params": {
        "latent_dim": 5,
        "view_latent_dim": 10,
        "encoder_struct": [[128, 64], [128, 64]],
        "decoder_struct": [[64, 128], [64, 128]],
        "epochs": 200,
        "lr": 1e-3,
        "batch_size": 32,
        "lambda_r": 1.0,
        "lambda_ent": 0.05,
        "lambda_sp": 0.05,
        "temperature": 0.7
    },
    "mcfl_params": {"n_clusters": 3, "gamma": 0.1, "max_iter": 20},
    "mrag_params": {"n_clusters": 3, "k_neighbors": 5},
    "nsgl_params": {"n_clusters": 3, "alpha": 0.1}
}

if __name__ == "__main__":
    print(f"Starting {CONFIG['task_name']} (Repeats: {CONFIG['n_repeats']}, Cores: {CONFIG['n_cores']})...")
    print(f"Config: Dims={CONFIG['dims']}, Samples={CONFIG['n_samples']}")
    print("-" * 60)

    task = partial(run_simulation_task_clu, config=CONFIG)

    if CONFIG["n_cores"] > 1:
        with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
            results = list(tqdm(pool.imap(task, range(CONFIG['n_repeats'])), total=CONFIG['n_repeats']))
    else:
        results = [task(s) for s in range(CONFIG['n_repeats'])]

    df = pd.DataFrame(results)

    # =========================================================================
    # Generate Detailed Report (Matches Classification Style)
    # =========================================================================
    print("\n" + "=" * 120)
    print(f" >>> Final Clustering Performance (Mean(SD)) over {CONFIG['n_repeats']} runs")
    print("=" * 120)

    # 聚类任务的方法列表 (无监督一般没有 Validation Best 概念，只取 Final)
    methods = ["MDFS_Final", "MCFL", "MRAG", "NSGL"]

    # 1. 动态识别所有相关的指标列 (Total + Per-View)
    # 查找所有包含 MDFS_Final 且包含 recall 或 precision 的列
    all_metric_cols = [c for c in df.columns if ('recall' in c or 'precision' in c) and 'MDFS_Final' in c]

    # 去掉前缀 'MDFS_Final_' 以获取通用的指标名
    base_metric_names = sorted(list(set([c.replace('MDFS_Final_', '') for c in all_metric_cols])))


    # 排序优化：把 'total' 放在前面，其他按字母顺序
    def sort_key(name):
        if 'total' in name: return '0_' + name  # Ensure total comes first
        return '1_' + name


    base_metric_names.sort(key=sort_key)

    summary_rows = []

    for method in methods:
        row_data = {"Method": method}
        for base_metric in base_metric_names:
            col_name = f"{method}_{base_metric}"

            if col_name in df.columns:
                mean_val = df[col_name].mean()
                sd_val = df[col_name].std()
                # 格式化输出: Mean(SD)
                row_data[base_metric] = f"{mean_val:.4f}({sd_val:.4f})"
            else:
                row_data[base_metric] = "N/A"
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)

    # 调整列顺序: Method 在第一列
    if not summary_df.empty:
        cols = ["Method"] + [c for c in summary_df.columns if c != "Method"]
        summary_df = summary_df[cols]

    # 设置 Pandas 显示参数以完整打印表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    print(summary_df.to_string(index=False))

    # 保存结果
    save_path = os.path.join(project_root, "results", "sim_3_clu_summary.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)

    # 同时也保存原始数据
    raw_path = os.path.join(project_root, "results", "sim_3_clu_raw.csv")
    df.to_csv(raw_path, index=False)

    print(f"\nReport saved to {save_path}")