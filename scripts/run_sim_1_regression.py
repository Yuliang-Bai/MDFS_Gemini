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
    "n_repeats": 20, "n_cores": 4, "n_samples": 200, "dims": [300, 300],
    "noise": 0.5, "gamma": 0.7,
    "mdfs_params": {
        "latent_dim": 5, "view_latent_dim": 10,
        "encoder_struct": [[128, 64], [128, 64]],
        "decoder_struct": [[64, 128], [64, 128]],
        "temperature": 0.5, "epochs": 600, "lr": 1e-3,
        "lambda_r": 3, "lambda_ent": 0.05, "lambda_sp": 0.5
    },
    "lasso_params": {}, "slrfs_params": {}
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

    # ==========================================
    # 数据汇总与格式化 (Formatting Report)
    # ==========================================
    print("\n" + "=" * 80)
    print(" >>> Final Performance Report (Mean ± SD)")
    print("=" * 80)

    # 定义要展示的方法 (Rows)
    methods = ["MDFS_Best", "MDFS_Final", "AdaCoop", "MSGLasso", "SLRFS"]

    # 定义要展示的指标 (Columns)
    # 假设有两个模态 view1, view2
    metrics_to_show = [
        "recall_total", "precision_total",
        "recall_view1", "precision_view1",
        "recall_view2", "precision_view2"
    ]

    summary_rows = []

    for method in methods:
        row_data = {"Method": method}
        for metric in metrics_to_show:
            col_name = f"{method}_{metric}"

            # 检查列是否存在 (防止基准方法没有某些指标)
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

    # 设置显示格式，保留4位小数
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # 打印 MDFS 最佳 Epoch 统计
    if "MDFS_Best_Epoch" in df.columns:
        avg_epoch = df["MDFS_Best_Epoch"].mean()
        std_epoch = df["MDFS_Best_Epoch"].std()
        print(f"\n[Info] MDFS Average Best Epoch: {avg_epoch:.2f} ± {std_epoch:.2f}\n")

    print(summary_df.set_index("Method"))

    # 保存结果
    save_path = os.path.join(project_root, "results", "sim_1_reg_summary.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"\nReport saved to {save_path}")