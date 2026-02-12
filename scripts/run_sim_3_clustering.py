import sys
import os
import multiprocessing
from functools import partial

# --- Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

from src.utils.parallel import configure_for_multiprocessing, worker_init

# --- Configuration ---
CONFIG = {
    "task_name": "Clustering_Simulation",
    "n_repeats": 20,
    "n_cores": 4,
    "n_samples": 200,
    "dims": [100, 100],
    "n_clusters": 3,
    "mdfs_params": {
        "latent_dim": 5,
        "view_latent_dim": 10,
        "encoder_struct": [[128, 64], [128, 64]],
        "decoder_struct": [[64, 128], [64, 128]],
        "epochs": 600,
        "lr": 5e-2,
        "lambda_r": 1.0,
        "lambda_ent": 0.05,
        "lambda_sp": 0.05,
        "temperature": 0.7
    },
    "mcfl_params": {"n_clusters": 3, "gamma": 0.1, "max_iter": 20},
    "mrag_params": {"n_clusters": 3, "k_neighbors": 5},
    "nsgl_params": {"n_clusters": 3, "alpha": 0.1}
}

configure_for_multiprocessing(CONFIG["n_cores"], inner_threads=1)

import pandas as pd
from tqdm import tqdm
from src.simulations.clustering import run_simulation_task_clu

# ==========================================
# 3. 主程序 (Main Execution)
# ==========================================
if __name__ == "__main__":
    print(f"Starting {CONFIG['task_name']} (Repeats: {CONFIG['n_repeats']}, Cores: {CONFIG['n_cores']})...")
    print(f"Config: Dims={CONFIG['dims']}, Samples={CONFIG['n_samples']}")
    print("-" * 60)

    # 1. 准备并行任务
    task = partial(run_simulation_task_clu, config=CONFIG)

    # 2. 执行并行计算 (带进度条)
    if CONFIG["n_cores"] > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(CONFIG["n_cores"], initializer=worker_init) as pool:
            # imap + tqdm 实现进度条
            results = list(tqdm(pool.imap(task, range(CONFIG['n_repeats'])), total=CONFIG['n_repeats']))
    else:
        # 单核调试模式
        results = [task(s) for s in tqdm(range(CONFIG['n_repeats']))]

    # 3. 结果转换为 DataFrame
    df = pd.DataFrame(results)

    # ==========================================
    # 4. 生成最终报告 (Mean ± SD)
    # ==========================================
    print("\n" + "=" * 100)
    print(f" >>> Final Clustering Performance (Mean(SD)) over {CONFIG['n_repeats']} runs")
    print("=" * 100)

    # 定义要展示的方法 (包含 MDFS_Best)
    methods = ["MDFS_Best", "MDFS_Final", "MCFL", "MRAG", "NSGL"]

    # 动态识别指标列 (以 MDFS_Final 为基准，查找 recall/precision 相关列)
    all_metric_cols = [c for c in df.columns if ('recall' in c or 'precision' in c) and 'MDFS_Final' in c]
    # 去掉前缀，获取基础指标名 (e.g., 'recall_total', 'view1_precision')
    base_metric_names = sorted(list(set([c.replace('MDFS_Final_', '') for c in all_metric_cols])))


    # 排序优化：确保 'total' 指标排在 'view' 指标前面
    def sort_key(name):
        if 'total' in name: return '0_' + name
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
                # 格式化: Mean(SD)
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

    # 5. 保存结果
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 保存汇总表格
    save_path = os.path.join(results_dir, "sim_3_clu_summary.csv")
    summary_df.to_csv(save_path, index=False)

    # 保存原始数据 (Raw)
    raw_path = os.path.join(results_dir, "sim_3_clu_raw.csv")
    df.to_csv(raw_path, index=False)

    print(f"\nReport saved to: {save_path}")