import sys
import os
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# 0. 修复 Jupyter/IPython 环境路径问题
# ==========================================
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # 交互式环境 (Jupyter/IPython) 下的 Fallback
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(
        current_dir) == 'scripts' else current_dir
if project_root not in sys.path: sys.path.append(project_root)

# 导入所有方法
from src.methods.classification.proposed import MDFSClassifier
from src.methods.classification.baselines import SMVFS, HLRFS, SCFS


def main():
    # ---------------- 1. 加载并处理数据 ----------------
    print("正在加载数据...")
    data = scipy.io.loadmat('data/bbcsport.mat')

    X_mat = data['X']
    view1 = X_mat[0, 0]
    view2 = X_mat[1, 0]

    # 将稀疏矩阵转换为稠密矩阵 (MDFS 和 Baselines 需要 ndarray)
    if scipy.sparse.issparse(view1):
        view1 = view1.toarray()
    if scipy.sparse.issparse(view2):
        view2 = view2.toarray()

    # 处理标签 y (降为 1 维，并强制从 0 开始连续编码)
    y = data['y'].flatten()
    le = LabelEncoder()
    y = le.fit_transform(y)

    n_samples = len(y)
    n_classes = len(np.unique(y))
    print(f"数据加载完成: 样本数={n_samples}, 类别数={n_classes}")
    print(f"View 1 特征数: {view1.shape[1]}, View 2 特征数: {view2.shape[1]}")

    # ---------------- 2. 初始化参数 (同步自分类模拟配置) ----------------
    # MDFS 参数
    mdfs_params = {
        "n_classes": n_classes,
        "latent_dim": 5,
        "view_latent_dim": 10,
        "encoder_struct": [[128, 64], [128, 64]],
        "decoder_struct": [[64, 128], [64, 128]],
        "temperature": 0.5,
        "epochs": 100,
        "lr": 5e-3,
        "lambda_r": 5,
        "lambda_ent": 0.05,
        "lambda_sp": 0.5
    }

    # Baselines 参数
    smvfs_params = {"alpha": 100, "rho": 0.1, "mu": 100, "max_iter": 100}
    hlrfs_params = {"beta": 1, "gamma": 100, "n_neighbors": 10, "r": 10, "max_iter": 100}
    scfs_params = {"lambda1": 100, "lambda2": 1, "rho": 10, "max_iter": 100}

    methods_config = [
        ("MDFS", MDFSClassifier, mdfs_params),
        ("SMVFS", SMVFS, smvfs_params),
        ("HLRFS", HLRFS, hlrfs_params),
        ("SCFS", SCFS, scfs_params)
    ]

    # 初始化存放准确率的字典
    results_acc = {name: [] for name, _, _ in methods_config}

    # ---------------- 3. 开始 10 折交叉验证 ----------------
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(view1, y)):
        print(f"\n========== 正在处理第 {fold + 1}/10 折 ==========")

        # a. 划分数据 (包装成字典)
        X_train = {"view1": view1[train_index], "view2": view2[train_index]}
        X_test = {"view1": view1[test_index], "view2": view2[test_index]}
        y_train, y_test = y[train_index], y[test_index]

        # b. 遍历各个方法
        for name, ModelClass, params in methods_config:
            print(f"  -> 正在运行 {name}...")
            try:
                # 训练模型获取筛选特征
                model = ModelClass(**params)
                # 针对 MDFS 关闭 verbose 防止刷屏
                if name == "MDFS":
                    fit_result = model.fit(X_train, y_train, verbose=False)
                else:
                    fit_result = model.fit(X_train, y_train)

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

                # === 数据标准化 ===
                # 严格使用训练集拟合 StandardScaler，再转换训练集和测试集
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_concat)
                X_test_scaled = scaler.transform(X_test_concat)

                # 训练 SVM (支持多分类，默认 linear 核表现较好)
                svm = SVC(kernel='linear', C=1.0, random_state=42)
                svm.fit(X_train_scaled, y_train)
                y_pred = svm.predict(X_test_scaled)

                # 计算并记录准确率
                acc = accuracy_score(y_test, y_pred)
                results_acc[name].append(acc)
                print(f"     [+] {name} 筛选完成, 当前折 SVM Accuracy: {acc:.4f}")

            except Exception as e:
                print(f"     [-] {name} 运行出错: {e}")
                results_acc[name].append(np.nan)  # 发生异常时填充 NaN

    # ---------------- 4. 汇总并打印结果 ----------------
    print("\n\n" + "=" * 60)
    print(" >>> 实例分析最终结果 (10折交叉验证 - BBCSport)")
    print("=" * 60)

    summary_data = []
    for name in results_acc.keys():
        acc_list = np.array(results_acc[name])
        # 忽略由于报错产生的 NaN
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
            "Valid Folds": f"{valid_folds}/10"
        })

    summary_df = pd.DataFrame(summary_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()