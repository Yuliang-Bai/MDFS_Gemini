import os
import sys
import numpy as np
from copy import deepcopy
from ...base import BaseMethod, FitResult


# ==============================================================================
# 【环境核心修复】 强制配置 Conda 内部 R 环境
# ==============================================================================
def setup_r_environment():
    """
    在导入 rpy2 之前，强制将当前 Conda 环境的 R 路径注入到环境变量中。
    解决 'DLL Hell' 和 '.Internal(gettext)' 版本不匹配错误。
    """
    # 1. 获取当前 Conda 环境根目录 (例如 D:\Miniconda3\envs\r_msg)
    conda_root = sys.prefix

    # 2. 推导 R 的关键路径
    # Conda R 通常位于 %PREFIX%\lib\R
    r_home = os.path.join(conda_root, 'lib', 'R')
    # R.dll 通常位于 %PREFIX%\lib\R\bin\x64
    r_bin_x64 = os.path.join(r_home, 'bin', 'x64')

    # 3. 验证路径是否存在
    if not os.path.exists(r_home):
        print(f"[Warning] 无法在当前环境中找到 R_HOME: {r_home}")
        return False

    # 4. 暴力设置环境变量 (覆盖系统设置)
    os.environ['R_HOME'] = r_home
    # Windows 下 rpy2 需要 R_USER，设为当前用户目录
    os.environ['R_USER'] = os.path.expanduser('~')

    # 5. 【最关键一步】将 R 的 bin 目录插到 PATH 的最前面
    # 确保加载的是 Conda 的 R.dll，而不是系统其他地方的旧版 DLL
    current_path = os.environ.get('PATH', '')
    if r_bin_x64 not in current_path:
        os.environ['PATH'] = r_bin_x64 + os.pathsep + current_path
        print(f"[Info] 已强制注入 R 路径到 PATH 头部: {r_bin_x64}")

    return True


# 执行环境配置
has_r_env = setup_r_environment()

# ==============================================================================
# 只有在环境配置完成后，才导入 rpy2 和 sklearn
# ==============================================================================
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    # 定义 R 包装器函数
    if has_r_env:
        r_wrapper_code = """
            run_msglasso_wrapper <- function(X, Y, group_ids) {
                library(MSGLasso)

                # ------------------------------------------------------------------
                # 1. 动态修复 R 包 Bug (Pen_L -> Pen.L)
                # ------------------------------------------------------------------
                fix_msglasso_bug <- function() {
                    # 获取源码
                    fun_text <- deparse(MSGLasso)
                    # 替换错误的变量名
                    fun_text <- gsub("Pen_L", "Pen.L", fun_text)
                    fun_text <- gsub("Pen_G", "Pen.G", fun_text)
                    # 重新解析
                    fixed_fun <- eval(parse(text = fun_text))
                    environment(fixed_fun) <- environment(MSGLasso)
                    return(fixed_fun)
                }
                # 获取修复后的函数
                MSGLasso_Fixed <- fix_msglasso_bug()

                # ------------------------------------------------------------------
                # 2. 数据准备
                # ------------------------------------------------------------------
                X <- as.matrix(X)
                Y <- as.matrix(Y)
                p <- ncol(X)
                q <- ncol(Y)

                # 组结构
                groups <- unique(group_ids)
                G <- length(groups)

                G.Starts <- c()
                G.Ends <- c()
                curr <- 1
                for (g in groups) {
                    len <- sum(group_ids == g)
                    G.Starts <- c(G.Starts, curr)
                    G.Ends <- c(G.Ends, curr + len - 1)
                    curr <- curr + len
                }

                # 响应变量结构 (假设1组)
                R_num <- 1
                R.Starts <- c(1)
                R.Ends <- c(q)

                # 辅助矩阵计算
                gmax <- 1 
                cmax <- max(c(G.Ends - G.Starts + 1, R.Ends - R.Starts + 1))

                pq <- FindingPQGrps(p, q, G, R_num, gmax, G.Starts, G.Ends, R.Starts, R.Ends)
                PQ.grps <- pq$PQgrps

                gr <- FindingGRGrps(p, q, G, R_num, cmax, G.Starts, G.Ends, R.Starts, R.Ends)
                GR.grps <- gr$GRgrps

                wts <- Cal_grpWTs(p, q, G, R_num, gmax, PQ.grps)
                grp.WTs <- wts$grpWTs

                # 惩罚项矩阵
                Pen.L <- matrix(1, nrow=p, ncol=q)
                Pen.G <- matrix(1, nrow=G, ncol=R_num)

                # Lambda
                lam1 <- 0.1
                lam.G <- matrix(0.1, nrow=G, ncol=R_num)

                # 【新增修复】 初始化 grp_Norm0
                # 这是一个必传参数，表示 Beta0 的组范数。因为 Beta0 默认是 0，所以这里也全是 0
                grp_Norm0 <- matrix(0, nrow=G, ncol=R_num)

                # ------------------------------------------------------------------
                # 3. 执行 (传入所有必填参数)
                # ------------------------------------------------------------------
                res <- MSGLasso_Fixed(X.m=X, Y.m=Y, grp.WTs=grp.WTs, 
                                      Pen.L=Pen.L, Pen.G=Pen.G, 
                                      PQ.grps=PQ.grps, GR.grps=GR.grps, 
                                      grp_Norm0=grp_Norm0,   # <--- 之前缺了这个
                                      lam1=lam1, lam.G=lam.G)

                return(res$Beta)
            }
        """
        try:
            robjects.r(r_wrapper_code)
        except Exception as e:
            print(f"Warning: Failed to define R wrapper. Error: {e}")

except ImportError:
    pass

from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from scipy.linalg import eigh, inv, norm

class AdaCoop(BaseMethod):
    """
    Implementation of Cooperative Learning (Algorithm 1) from Ding et al. (2022).
    This constructs an augmented matrix to enforce agreement between views.
    """

    def __init__(self, name="AdaCoop", rho_grid=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # 默认搜索的 rho 值，包含 0 (Early Fusion), 1 (Late Fusion 近似), 以及其他强度
        #  建议在一个 grid 上搜索 rho
        self.rho_grid = rho_grid if rho_grid is not None else [0, 0.1, 0.5, 1.0, 5.0, 10.0]

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        # 1. 预处理：确保只有两个视图 (论文主要推导基于双视图 X 和 Z [cite: 106])
        sorted_keys = sorted(X.keys())
        if len(sorted_keys) != 2:
            # 如果超过2个视图，论文建议使用迭代算法或成对约束 [cite: 583]，
            # 这里为保持代码简洁，暂时只处理前两个视图，实际应用需扩展
            print(f"Warning: AdaCoop expects 2 views, found {len(sorted_keys)}. Using first two.")
            sorted_keys = sorted_keys[:2]

        view1_name, view2_name = sorted_keys
        X1 = X[view1_name]
        X2 = X[view2_name]

        # 标准化是必须的，假设输入的X和Z列已经标准化 [cite: 131]
        # 这里为了安全再次进行标准化处理
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        X1_std = scaler1.fit_transform(X1)
        X2_std = scaler2.fit_transform(X2)

        n_samples = y.shape[0]
        p1 = X1_std.shape[1]
        p2 = X2_std.shape[1]

        # 2. 交叉验证选择最佳 rho [cite: 170]
        # Remark A: CV 必须基于原始 X/Z 的行进行划分，而不是增广后的矩阵
        best_rho = 0
        best_mse = float('inf')

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for rho in self.rho_grid:
            cv_errors = []

            for train_idx, val_idx in kf.split(X1_std):
                # 切分数据
                X1_train, X1_val = X1_std[train_idx], X1_std[val_idx]
                X2_train, X2_val = X2_std[train_idx], X2_std[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 构造增广矩阵 (Augmented Data) 用于训练 [cite: 148, 168]
                # X_tilde = [ X      Z     ]
                #           [ -sq_r*X sq_r*Z]
                # y_tilde = [ y ]
                #           [ 0 ]
                sqrt_rho = np.sqrt(rho)
                n_train = len(y_train)

                # 上半部分: 预测误差项
                top_X = np.hstack([X1_train, X2_train])
                top_y = y_train

                # 下半部分: 一致性惩罚项 (Agreement Penalty)
                # 只有 rho > 0 时才需要添加约束行
                if rho > 0:
                    bottom_X = np.hstack([-sqrt_rho * X1_train, sqrt_rho * X2_train])
                    bottom_y = np.zeros(n_train)

                    X_aug = np.vstack([top_X, bottom_X])
                    y_aug = np.concatenate([top_y, bottom_y])
                else:
                    X_aug = top_X
                    y_aug = top_y

                # 使用 Lasso/ElasticNet 求解问题 [cite: 150]
                # 这里使用固定的小 alpha 或简单的 CV 来加速内部循环
                model = Lasso(alpha=0.1, max_iter=5000)  # 简化：实际应嵌套 CV 选 lambda
                model.fit(X_aug, y_aug)

                # 预测：在验证集上不使用增广矩阵，直接预测 [cite: 137]
                # f(X) + f(Z) = X_beta_x + Z_beta_z
                X_val_concat = np.hstack([X1_val, X2_val])
                y_pred = model.predict(X_val_concat)

                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)

            avg_mse = np.mean(cv_errors)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_rho = rho

        # 3. 使用最佳 rho 在全部数据上重新训练 [cite: 170]
        sqrt_rho = np.sqrt(best_rho)

        top_X = np.hstack([X1_std, X2_std])
        top_y = y

        if best_rho > 0:
            bottom_X = np.hstack([-sqrt_rho * X1_std, sqrt_rho * X2_std])
            bottom_y = np.zeros(n_samples)
            X_final = np.vstack([top_X, bottom_X])
            y_final = np.concatenate([top_y, bottom_y])
        else:
            X_final = top_X
            y_final = top_y

        # 最终模型训练，这里可以使用 ElasticNetCV 来自动选择 lambda [cite: 153]
        final_model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=5,
            random_state=42,
            n_jobs=1,
            max_iter=5000
        )
        final_model.fit(X_final, y_final)

        # 4. 特征筛选逻辑 (保持与原代码一致)
        # 系数被拼接在一起了： [theta_x, theta_z]
        coefs = np.abs(final_model.coef_)

        # 简单的 Top-K 筛选策略
        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1
        top_idx = np.argsort(coefs)[::-1][:k_total]

        selected = {}
        curr = 0

        # 映射回原始视图 (X 和 Z)
        # 注意顺序：sorted_keys = [view1, view2]
        dims = [p1, p2]
        for i, name in enumerate(sorted_keys):
            dim = dims[i]
            # 找到属于当前视图的特征索引
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class MSGLasso(BaseMethod):
    """
    Baseline: Multivariate Sparse Group Lasso (Wrapper for R package 'MSGLasso')
    """

    def __init__(self, name="MSGLasso", **kwargs):
        super().__init__(name=name, **kwargs)
        self.r_available = has_r_env

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        if not self.r_available:
            return FitResult(selected_features={k: [] for k in X.keys()})

        # 1. 准备数据
        sorted_keys = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in sorted_keys])

        # 构造组索引向量 (Python List)
        # 例如: [1, 1, 1, 2, 2, 2, ...]
        group_ids = []
        feature_counts = []
        for i, key in enumerate(sorted_keys):
            n_cols = X[key].shape[1]
            feature_counts.append(n_cols)
            # R 是 1-based 索引，所以组号从 1 开始
            group_ids.extend([i + 1] * n_cols)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 2. 调用 R 包装器
        coef_matrix = None
        try:
            with localconverter(robjects.default_converter + numpy2ri.converter):
                # 调用我们在上面定义的 R 函数 run_msglasso_wrapper
                # 传入 X, Y 和 group_ids
                # 注意: group_ids 会被自动转为 IntVector
                r_group_ids = robjects.IntVector(group_ids)

                # 调用 R 函数
                result_beta = robjects.r['run_msglasso_wrapper'](X_concat, y, r_group_ids)
                coef_matrix = np.array(result_beta)

        except Exception as e:
            print(f"Error executing MSGLasso R wrapper: {e}")
            return FitResult(selected_features={k: [] for k in X.keys()})

        # 3. 处理结果
        # Beta 可能是 (p, q) 矩阵 (如果没有 path) 或者 (p, q, nlambda)
        if coef_matrix is not None:
            if coef_matrix.ndim == 3:
                # 如果返回了路径，取中间值
                idx = coef_matrix.shape[2] // 2
                final_beta = coef_matrix[:, :, idx]
            else:
                final_beta = coef_matrix

            feature_scores = np.linalg.norm(final_beta, axis=1)
        else:
            feature_scores = np.zeros(X_concat.shape[1])

        # 4. 筛选特征
        n = len(y)
        k_total = int(n / np.log(n)) if n > 1 else 1
        top_idx = np.argsort(feature_scores)[::-1][:k_total]

        selected = {}
        curr = 0
        for i, name in enumerate(sorted_keys):
            dim = feature_counts[i]
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class SLRFS(BaseMethod):
    """
    Method: Sparse Low-Rank Feature Selection (SLR-FS)
    Reference: Hu et al., "Low-rank feature selection for multi-view regression", 2017.

    Implements Algorithm 1 from the paper:
    Minimizes ||Y - X^T A B||_F^2 + lambda * ||A B||_{2,p}
    """

    def __init__(self, name="SLRFS", r=5, p=1.0, lambda_=1.0, max_iter=100, tol=1e-5, **kwargs):
        """
        Args:
            r (int): Rank of the low-rank constraint (subspace dimension).
            p (float): Parameter for l2,p norm (0 < p < 2). Default is 1.0 for sparsity.
            lambda_ (float): Regularization parameter.
            max_iter (int): Maximum iterations for the optimization loop.
            tol (float): Convergence tolerance.
        """
        super().__init__(name=name, **kwargs)
        self.r = r
        self.p = p
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: dict, y: np.ndarray) -> FitResult:
        """
        Fit the SLR-FS model for each view independently (as per Eq 7 in the paper).
        """
        # 1. 预处理 Y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 标准化 Y (论文中提到 normalized class indicator matrix)
        scaler_y = StandardScaler()
        y_std = scaler_y.fit_transform(y)

        n_samples, n_classes = y_std.shape

        # 确定筛选数量
        k_total = int(n_samples / np.log(n_samples)) if n_samples > 1 else 1

        selected_features = {}

        # 论文 Eq (7) 是对每个视图 v 分别优化的
        for view_name, X_v in X.items():
            # X_v: (n_samples, n_features)
            # 论文中的 X 是 (n_features, n_samples)，所以涉及 X 的公式需要转置处理

            # 标准化 X
            scaler_x = StandardScaler()
            X_std = scaler_x.fit_transform(X_v)

            # 转换为论文符号: X_paper (d x n), Y_paper (n x k)
            # 代码中保持 sklearn 习惯 (n x d)，计算时做相应转置
            # X (n, d)
            n, d = X_std.shape

            # 限制秩 r 不能超过特征数或类别数
            rank_r = min(self.r, d, n_classes)

            # --- Algorithm 1 ---

            # 1. Initialization
            # Random A (d, r), B (r, k)
            np.random.seed(42)
            A = np.random.randn(d, rank_r)
            B = np.random.randn(rank_r, n_classes)
            # D (d, d) Diagonal matrix initialized to Identity
            d_diag = np.ones(d)

            prev_obj = float('inf')

            for t in range(self.max_iter):
                # ---------------------------------------------------
                # Step 1: Update A (Subspace Matrix) via Eq (14)
                # Maximize Tr( (A^T (S_t + lambda D) A)^-1 A^T S_b A )
                # This is a Generalized Eigenvalue Problem: S_b v = val (S_t + lambda D) v
                # ---------------------------------------------------

                # S_t = X X^T (paper) -> X.T @ X (here)
                S_t = X_std.T @ X_std

                # S_b = X Y Y^T X^T (paper) -> X.T @ Y @ Y.T @ X (here)
                # 为了计算高效，先算 M = X.T @ Y
                M = X_std.T @ y_std
                S_b = M @ M.T

                # Regularized Total Scatter: S_total = S_t + lambda * D
                # D is diagonal
                S_total = S_t + np.diag(self.lambda_ * d_diag)

                # Solve generalized eigenvalue problem: S_b * v = w * S_total * v
                # We need top r eigenvectors
                try:
                    vals, vecs = eigh(S_b, S_total, subset_by_index=(d - rank_r, d - 1))
                    # eigh 返回的是升序，取最后 r 个，并反转顺序
                    A_new = np.fliplr(vecs)
                except Exception:
                    # 如果 S_total 奇异，退化为普通特征值分解或加抖动
                    vals, vecs = eigh(S_b)
                    A_new = np.fliplr(vecs)[:, :rank_r]

                A = A_new

                # ---------------------------------------------------
                # Step 2: Update B (Coefficient Matrix) via Eq (12)
                # B = (A^T (X X^T + lambda D) A)^-1 A^T X Y
                # ---------------------------------------------------
                term1 = A.T @ S_total @ A
                term2 = A.T @ M  # M = X.T @ Y

                # Add small regularization to inversion for stability
                term1_inv = inv(term1 + 1e-6 * np.eye(rank_r))
                B = term1_inv @ term2

                # ---------------------------------------------------
                # Step 3: Update D (Weight Matrix) via Eq (8)
                # d_ii = 1 / ( (2/p) * ||z^i||_2^(2-p) )
                # z^i is the i-th row of Z = A @ B
                # ---------------------------------------------------
                Z = A @ B  # (d, k)
                row_norms = norm(Z, axis=1)

                # Avoid division by zero
                row_norms = np.maximum(row_norms, 1e-8)

                # Eq (8): d_ii = p / (2 * ||z||^(2-p))
                # derived from paper: d_ii = 1 / ( (2/p) * ... )
                exponent = 2.0 - self.p
                d_diag = (self.p / 2.0) * (row_norms ** (self.p - 2.0))

                # ---------------------------------------------------
                # Check Convergence (Eq 7)
                # Obj = ||Y - X^T A B||^2 + lambda * ||A B||_{2,p}
                # ---------------------------------------------------
                # Residual
                Y_pred = X_std @ Z
                loss = np.sum((y_std - Y_pred) ** 2)

                # Regularization: sum( ||z^i||^p )
                reg = np.sum(row_norms ** self.p)

                obj = loss + self.lambda_ * reg

                if abs(prev_obj - obj) < self.tol:
                    break
                prev_obj = obj

            # --- Feature Selection ---
            # 根据最终的系数矩阵 Z = AB 的行范数进行排序
            # Row-sparsity: 重要的特征对应的行范数大，不重要的趋近于 0
            Z_final = A @ B
            feat_scores = norm(Z_final, axis=1)

            # Select Top-K
            # argsort 是升序，[::-1] 转降序
            idx = np.argsort(feat_scores)[::-1][:k_total]
            selected_features[view_name] = idx.tolist()

        return FitResult(selected_features=selected_features)