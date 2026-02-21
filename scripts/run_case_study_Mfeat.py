import sys
import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
# ==========================================
# 0. 修复环境路径 & 引入并行优化
# ==========================================
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

from src.utils.parallel import configure_for_multiprocessing

configure_for_multiprocessing(n_cores=10, inner_threads=1)

from src.methods.case_study.case_study_proposed import MDFSClassifier
from src.methods.case_study.case_study_baselines import SMVFS, HLRFS, SCFS


# ==========================================
# 独立的工作函数：用于处理单次任务
# ==========================================
def run_single_task(task_id, train_index, test_index, view1, view2, y, methods_config):
    # task_id 格式如 "Rep1-Fold5"
    print(f"[Worker] -> 开始处理 {task_id} ...")

    X_train = {"view1": view1[train_index], "view2": view2[train_index]}
    X_test = {"view1": view1[test_index], "view2": view2[test_index]}
    y_train, y_test = y[train_index], y[test_index]

    # 【稠密特征核心预处理】：StandardScaler
    scaler_v1 = StandardScaler()
    scaler_v2 = StandardScaler()

    X_train["view1"] = scaler_v1.fit_transform(X_train["view1"])
    X_test["view1"] = scaler_v1.transform(X_test["view1"])

    X_train["view2"] = scaler_v2.fit_transform(X_train["view2"])
    X_test["view2"] = scaler_v2.transform(X_test["view2"])

    fold_results = {}

    for name, ModelClass, params in methods_config:
        try:
            model = ModelClass(**params)
            # 训练模型获取筛选特征 (MDFS 关闭 verbose)
            fit_result = model.fit(X_train, y_train, verbose=False) if name == "MDFS" else model.fit(X_train, y_train)

            selected_features = fit_result.selected_features

            # 提取筛选出的特征并拼接
            X_train_concat = np.hstack((
                X_train["view1"][:, selected_features["view1"]],
                X_train["view2"][:, selected_features["view2"]]
            ))
            X_test_concat = np.hstack((
                X_test["view1"][:, selected_features["view1"]],
                X_test["view2"][:, selected_features["view2"]]
            ))

            # 训练 SVM
            svm = SVC(kernel='linear', C=1.0, random_state=42)
            svm.fit(X_train_concat, y_train)
            y_pred = svm.predict(X_test_concat)

            # 计算准确率
            acc = accuracy_score(y_test, y_pred)
            fold_results[name] = acc
            print(f"   [{task_id}] {name} 筛选完成, Accuracy: {acc:.4f}")

        except Exception as e:
            print(f"   [{task_id}] {name} 运行出错: {e}")
            fold_results[name] = np.nan

    return task_id, fold_results


def main():
    # ---------------- 1. 加载并处理数据 ----------------
    print("正在加载 Handwritten 数据...")
    data = scipy.io.loadmat('data/Handwritten.mat')

    # 提取 View 1 (子视图1: 索引0) 和 View 5 (子视图5: 索引4)
    X_mat = data['X']
    view1 = X_mat[0, 0].astype(np.float32)  # 216 维 轮廓相关性
    view2 = X_mat[4, 0].astype(np.float32)  # 240 维 像素平均值

    y = data['y'].flatten()
    y = LabelEncoder().fit_transform(y)

    n_samples_total, n_classes = len(y), len(np.unique(y))
    print(f"数据整体加载完成: 总样本数={n_samples_total}, 类别数={n_classes}")

    # ---------------- 2. 初始化参数 (针对极小样本+稠密特征优化) ----------------
    mdfs_params = {
        "n_classes": n_classes,
        "latent_dim": 5,
        "view_latent_dim": 16,
        "encoder_struct": [[64], [64]],
        "decoder_struct": [[64], [64]],
        "dropout": 0.3,
        "temperature": 0.5,
        "epochs": 400,
        "lr": 5e-3,
        "lambda_r": 1.0,
        "lambda_ent": 0.05,
        "lambda_sp": 0.5
    }

    smvfs_params = {"alpha": 100, "rho": 0.1, "mu": 100, "max_iter": 100}
    hlrfs_params = {"beta": 1, "gamma": 100, "n_neighbors": 10, "r": 10, "max_iter": 10}
    scfs_params = {"lambda1": 100, "lambda2": 1, "rho": 10, "max_iter": 15}

    methods_config = [
        ("MDFS", MDFSClassifier, mdfs_params),
        ("SMVFS", SMVFS, smvfs_params),
        ("HLRFS", HLRFS, hlrfs_params),
        ("SCFS", SCFS, scfs_params)
    ]

    # ---------------- 3. 生成 10次重复 × 10折交叉验证 的任务队列 ----------------
    tasks = []

    # 步骤 A：重复 10 次，每次分层随机抽取 200 个样本
    sss = StratifiedShuffleSplit(n_splits=10, train_size=300, random_state=42)

    for rep_idx, (sub_indices, _) in enumerate(sss.split(view1, y)):
        y_sub = y[sub_indices]  # 抽出来的 200 个样本的标签

        # 步骤 B：对这 200 个样本内部进行 10 折交叉验证
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rep_idx)

        for fold_idx, (train_idx_rel, test_idx_rel) in enumerate(kf.split(np.zeros(300), y_sub)):
            # 将 200 个样本内的相对索引 (0~199) 映射回全量数据集 (0~1999) 的绝对索引
            train_idx_abs = sub_indices[train_idx_rel]
            test_idx_abs = sub_indices[test_idx_rel]

            task_name = f"Rep{rep_idx + 1}-Fold{fold_idx + 1}"
            tasks.append((task_name, train_idx_abs, test_idx_abs))

    print(f"\n========== 启动并行验证 (共 {len(tasks)} 个任务, 10 进程并发) ==========")

    # 将 100 个任务全部送入并行池
    parallel_results = Parallel(n_jobs=10)(
        delayed(run_single_task)(
            task_name, train_idx, test_idx, view1, view2, y, methods_config
        ) for task_name, train_idx, test_idx in tasks
    )

    # ---------------- 4. 汇总 100 次运行的整体结果 ----------------
    results_acc = {name: [] for name, _, _ in methods_config}

    # 为了让打印好看一点，不用特意排序了，直接塞进去
    for _, fold_res in parallel_results:
        for name in results_acc.keys():
            results_acc[name].append(fold_res.get(name, np.nan))

    print("\n\n" + "=" * 60)
    print(" >>> 实例分析最终结果 (10次抽取200样本 × 10折交叉 - Handwritten)")
    print("=" * 60)

    summary_data = []
    for name in results_acc.keys():
        acc_list = np.array(results_acc[name])
        valid_acc = acc_list[~np.isnan(acc_list)]

        if len(valid_acc) > 0:
            mean_acc = np.mean(valid_acc)
            std_acc = np.std(valid_acc)
            valid_folds = len(valid_acc)
        else:
            mean_acc, std_acc, valid_folds = 0.0, 0.0, 0

        summary_data.append({
            "Method": name,
            "Accuracy (Mean ± Std)": f"{mean_acc:.4f} ± {std_acc:.4f}",
            "Valid Runs": f"{valid_folds}/{len(tasks)}"
        })

    summary_df = pd.DataFrame(summary_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()