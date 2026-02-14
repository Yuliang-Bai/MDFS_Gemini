import sys
import os
import multiprocessing

from functools import partial

# Path Setup
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

from src.utils.parallel import configure_for_multiprocessing, worker_init

CONFIG = {
    "n_repeats": 20, "n_cores": 20, "n_samples": 200, "dims": [300, 300],
    "noise": 1.0, "gamma": 0.7,
    "mdfs_params": {
        "latent_dim": 5, "view_latent_dim": 10,
        "encoder_struct": [[128, 64], [128, 64]],
        "decoder_struct": [[64, 128], [64, 128]],
        "temperature": 0.5, "epochs": 100, "lr": 1e-2,
        "lambda_r": 5, "lambda_ent": 0.05, "lambda_sp": 0.5
    },
    "ada_params": {},
    "msg_params": {},
    "slrfs_params": {"r": 5, "p": 1.0, "lambda": 1.0, "max_iter": 200, "tol": 1e-5}
}

configure_for_multiprocessing(CONFIG["n_cores"], inner_threads=1)

import pandas as pd
from tqdm import tqdm
from src.methods.regression.proposed import MDFSRegressor
from src.methods.regression.baselines import AdaCoop, MSGLasso, SLRFS
from src.simulations.regression import run_simulation_task

if __name__ == "__main__":
    print(f"Starting Simulation ({CONFIG['n_repeats']} repeats)...")
    print(f" - MDFS Config: {CONFIG['mdfs_params']['encoder_struct']}")
    print("-" * 50)

    # 运行模拟
    task = partial(run_simulation_task, config=CONFIG)
    if CONFIG["n_cores"] > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(CONFIG["n_cores"], initializer=worker_init) as pool:
            results = list(tqdm(pool.imap(task, range(CONFIG['n_repeats'])), total=CONFIG['n_repeats']))
    else:
        results = [task(s) for s in range(CONFIG['n_repeats'])]

    df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print(f" >>> Final Regression Performance (Mean(SD)) over {CONFIG['n_repeats']} runs")
    print("=" * 100)

    # 1. 定义方法列表
    methods = ["MDFS_Best", "MDFS_Final", "AdaCoop", "MSGLasso", "SLRFS"]

    # 2. 动态识别指标列 (逻辑与 Sim 2 完全一致)
    all_metric_cols = [c for c in df.columns if ('recall' in c or 'precision' in c) and 'MDFS_Final' in c]
    base_metric_names = sorted(list(set([c.replace('MDFS_Final_', '') for c in all_metric_cols])))


    # 排序：Total 在前，其他按字母
    def sort_key(name):
        if 'total' in name: return '0_' + name
        return '1_' + name


    base_metric_names.sort(key=sort_key)

    # 3. 构建汇总表
    summary_rows = []
    for method in methods:
        row_data = {"Method": method}
        for base_metric in base_metric_names:
            col_name = f"{method}_{base_metric}"
            if col_name in df.columns:
                mean_val = df[col_name].mean()
                sd_val = df[col_name].std()
                row_data[base_metric] = f"{mean_val:.4f}({sd_val:.4f})"
            else:
                row_data[base_metric] = "N/A"
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)

    # 调整列顺序
    if not summary_df.empty:
        cols = ["Method"] + [c for c in summary_df.columns if c != "Method"]
        summary_df = summary_df[cols]

    # 4. 打印与保存
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # 可选：打印 Epoch 信息
    if "MDFS_Best_Epoch" in df.columns:
        print(f"\n[Info] MDFS Avg Best Epoch: {df['MDFS_Best_Epoch'].mean():.1f} ± {df['MDFS_Best_Epoch'].std():.1f}\n")

    print(summary_df.to_string(index=False))

    save_path = os.path.join(project_root, "results", "sim_1_reg_summary.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"\nReport saved to {save_path}")