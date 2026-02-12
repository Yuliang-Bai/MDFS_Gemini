import numpy as np
from scipy.linalg import eigh, norm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional
from ...base import BaseMethod, FitResult


class MCFL(BaseMethod):
    """
    Baseline: MCFL (Multi-View Clustering and Feature Learning)
    Reference: Wang et al., "Multi-view clustering and feature learning via structured sparsity", 2013.

    Strict implementation of Algorithm 1.
    """

    def __init__(self, name="MCFL", n_clusters=3, gamma1=100.0, gamma2=100.0, max_iter=20, tol=1e-4, **kwargs):
        """
        Args:
            n_clusters (c): Number of clusters.
            gamma1 (float): Parameter for Group l1-norm (View sparsity).
            gamma2 (float): Parameter for l2,1-norm (Feature sparsity).
            max_iter (int): Maximum iterations.
            tol (float): Convergence tolerance.
        """
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: Dict[str, np.ndarray], y=None) -> FitResult:
        # ==========================================================
        # 1. Data Preparation & Notation Alignment
        # ==========================================================
        # Paper notation: X is d x n (columns are samples)
        # Input X dict: values are n x d_v

        sorted_keys = sorted(X.keys())
        X_list = []
        view_indices = []  # Stores (start, end) indices for each view in the concatenated feature dimension
        start = 0

        # Standardize and Concatenate
        for k in sorted_keys:
            # Paper assumes centered data (Eq. 5 derivation implies this)
            data_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X[k])
            X_list.append(data_std)

            dim = data_std.shape[1]
            view_indices.append((start, start + dim))
            start += dim

        X_concat = np.hstack(X_list)  # Shape: (n, d)
        X_paper = X_concat.T  # Shape: (d, n) -> Corresponds to 'X' in paper
        Xt_paper = X_concat  # Shape: (n, d) -> Corresponds to 'X^T' in paper

        d, n = X_paper.shape
        c = self.n_clusters

        # Precompute XX^T for the linear system (d x d)
        XXt = X_paper @ X_paper.T
        # Add small jitter for numerical stability of linear solver
        eye_d = np.eye(d) * 1e-6

        # ==========================================================
        # 2. Initialization (Step 1)
        # ==========================================================
        # "Initialize F_t by K-means clustering"
        kmeans = KMeans(n_clusters=c, n_init=10, random_state=42)
        kmeans.fit(Xt_paper)
        labels = kmeans.labels_

        # Construct F (n x c) from labels
        F = np.zeros((n, c))
        for i in range(n):
            F[i, labels[i]] = 1.0

        # "Initialize W_t and b_t"
        # Can be random or zeros. Paper implies solving Eq(2) first, but random start is common.
        W = np.random.randn(d, c) * 0.01
        b = np.zeros((c, 1))

        # ==========================================================
        # 3. Iterative Optimization (Steps 2-5)
        # ==========================================================
        for t in range(self.max_iter):
            W_prev = W.copy()

            # --- Step 2: Update F ---
            # "Calculate F_{t+1} = U[I;0]V^T where U and V are obtained by SVD on X^T W + 1 b^T"
            # M = X^T W + 1 b^T. Shape: (n, c)
            # Note: b.T is (1, c), broadcasted to (n, c)
            M = Xt_paper @ W + b.T

            # SVD: M = U Sigma V^T
            U, _, Vt = np.linalg.svd(M, full_matrices=False)

            # F = U V^T (The [I; 0] effectively selects the top c singular vectors, which thin SVD does)
            F = U @ Vt

            # --- Step 3: Update b ---
            # "Calculate b_{t+1} = F^T 1_n / n"
            # b shape: (c, 1)
            b = (F.T @ np.ones((n, 1))) / n

            # --- Step 4: Update W ---
            # Solve linear system for each column w_i (corresponding to cluster i)
            # Equation: (XX^T + gamma1 D^i + gamma2 D_tilde) w_i = X (f_i - b_i)

            # 4a. Calculate D_tilde (Diagonal matrix, depends on row norms of W)
            # "i-th diagonal element is 1 / (2 ||w^i||_2)"
            # row_norms shape: (d,)
            row_norms = norm(W, axis=1) + 1e-8
            d_tilde_diag = 0.5 / row_norms

            # 4b. Solve for each cluster i
            W_new = np.zeros_like(W)

            for i in range(c):
                # Construct D^i (Block Diagonal matrix, depends on view norms of w_i)
                # "j-th diagonal block of D^i is (1 / 2 ||w_i^j||_2) * I_j"
                d_i_diag = np.zeros(d)
                w_col = W[:, i]  # Current column w_i

                for (v_start, v_end) in view_indices:
                    # Extract part of w_i corresponding to view j
                    w_view = w_col[v_start:v_end]
                    view_norm = norm(w_view) + 1e-8
                    val = 0.5 / view_norm
                    # Fill the diagonal block with scalar val
                    d_i_diag[v_start:v_end] = val

                # LHS Matrix A = XX^T + Gamma1 * D^i + Gamma2 * D_tilde
                # Note: Adding diagonal arrays is efficient
                reg_diag = self.gamma1 * d_i_diag + self.gamma2 * d_tilde_diag
                A = XXt + np.diag(reg_diag) + eye_d

                # RHS Vector = X (f_i - b_i)
                # f_i is i-th column of F
                # b_i is i-th element of b
                # X_paper is (d, n), (f_i - b_i) is (n,)
                rhs = X_paper @ (F[:, i] - b[i])

                # Solve A w_i = rhs
                try:
                    w_sol = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    w_sol = np.linalg.lstsq(A, rhs, rcond=None)[0]

                W_new[:, i] = w_sol

            # --- Step 5: Convergence Check ---
            W = W_new
            diff = norm(W - W_prev) / (norm(W_prev) + 1e-8)
            if diff < self.tol:
                break

        # ==========================================================
        # 4. Feature Selection
        # ==========================================================
        # "Select features based on ||w^l||_2" (row norms of W)
        feat_scores = norm(W, axis=1)

        # Select Top-K
        k_total = max(1, int(n / np.log(n))) if n > 1 else 1
        top_idx = np.argsort(feat_scores)[::-1][:k_total]

        selected = {}
        curr = 0
        for k in sorted_keys:
            dim = X[k].shape[1]
            # Map global indices back to local view indices
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[k] = sel
            curr += dim

        return FitResult(selected_features=selected)


class MRAG(BaseMethod):
    """
    Verified Baseline: MRAG (Specifically the RHLC Feature Selection module)
    Reference: Jing et al., "Learning robust affinity graph representation for multi-view clustering", 2021.

    This implementation focuses on the 'Robust Hypergraph Laplacians Construction (RHLC)'
    phase (Section 3.1 of the paper), which selects features by jointly modelling
    hypergraph embedding and sparse regression.
    """

    def __init__(self, name="MRAG", n_clusters=3, k_neighbors=5, alpha=1.0, beta=0.1, max_iter=20, **kwargs):
        """
        Args:
            n_clusters: Number of clusters (dimension of the embedding Y).
            k_neighbors: Number of neighbors for hyperedge construction.
            alpha: Parameter weighting the regression loss term (||U^T X - Y||^2).
                   Matches the parameter before the loss term in Eq. (6).
            beta: Regularization parameter for L2,1 norm (||U||_2,1).
            max_iter: Maximum iterations for the IRLS solver.
        """
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def _construct_hypergraph_laplacian(self, X: np.ndarray) -> np.ndarray:
        """
        Construct Hypergraph Laplacian as per Eq (3) & (4) in the paper.
        """
        n_samples = X.shape[0]

        # 1. Find KNN for each sample (forming N hyperedges)
        # "Form a hyperedge by circling around its k-nearest neighbors" [cite: 1535]
        # We use k+1 to include the point itself + k neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X)
        _, indices = nbrs.kneighbors(X)

        # 2. Incidence Matrix H (N x N)
        # H_ij = 1 if vertex j is in hyperedge e_i (centered at i)
        H = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            H[indices[i], i] = 1.0

            # 3. Degrees
        # D_v: Vertex degree (sum of H rows)
        dv = np.sum(H, axis=1)
        # D_e: Hyperedge degree (sum of H cols)
        de = np.sum(H, axis=0)

        # 4. Compute Theta = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
        # Assuming W (hyperedge weights) is Identity as not specified otherwise for initialization
        dv_inv_sqrt = np.diag(1.0 / np.sqrt(dv + 1e-8))
        de_inv = np.diag(1.0 / (de + 1e-8))

        Theta = dv_inv_sqrt @ H @ de_inv @ H.T @ dv_inv_sqrt

        # 5. Laplacian L = I - Theta [cite: 1440]
        L = np.eye(n_samples) - Theta

        # Ensure symmetry for numerical stability
        L = (L + L.T) / 2.0
        return L

    def _solve_rhlc_view(self, X: np.ndarray) -> np.ndarray:
        """
        Solve Eq (6) & (7) for a single view.

        Objective: min alpha * ||X U - Y||_F^2 + beta * ||U||_2,1
        (Note: Code uses X (N x D), Paper uses X (D x N), so formulations are equivalent via transpose)

        Strategy:
        1. Fix Y as the Spectral Embedding of the Hypergraph Laplacian L.
           (This is the standard 'Spectral Regression' relaxation for Joint Embedding methods).
        2. Solve for U using Iteratively Reweighted Least Squares (IRLS).
        """
        n_samples, n_features = X.shape

        # Step 1: Construct Hypergraph Laplacian L
        L = self._construct_hypergraph_laplacian(X)

        # Step 2: Obtain Y via Eigen-decomposition of L (Spectral Embedding)
        # Corresponds to min tr(Y^T L Y) s.t. Y^T Y = I
        vals, vecs = eigh(L)

        # We take the eigenvectors corresponding to the smallest non-zero eigenvalues.
        # Indices 1 to n_clusters+1 (skipping the 0th trivial constant vector)
        if self.n_clusters < n_samples:
            Y = vecs[:, 1: self.n_clusters + 1]
        else:
            Y = vecs

        # Step 3: Solve for U using IRLS
        # Objective derivative wrt U set to 0:
        # alpha * 2 * X^T (X U - Y) + beta * 2 * D U = 0
        # (alpha * X^T X + beta * D) U = alpha * X^T Y

        U = np.zeros((n_features, Y.shape[1]))

        # Precompute constant parts
        XtX = X.T @ X
        XtY = X.T @ Y
        eye_d = np.eye(n_features)

        # Initial D (diagonal weight matrix for L2,1 norm)
        d_diag = np.ones(n_features)

        for t in range(self.max_iter):
            # Construct Regularization Matrix: beta * D
            reg_matrix = np.diag(self.beta * d_diag)

            # LHS: alpha * X^T X + beta * D
            A = self.alpha * XtX + reg_matrix + 1e-6 * eye_d

            # RHS: alpha * X^T Y
            rhs = self.alpha * XtY

            # Solve linear system A * U = rhs
            try:
                U = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(A, rhs, rcond=None)[0]

            # Update D based on new U rows
            # D_ii = 1 / (2 * ||u^i||_2) [cite: 1451]
            row_norms = norm(U, axis=1)
            d_diag = 0.5 / (row_norms + 1e-8)

        # Return Feature Scores (L2 norm of U rows)
        # "Calculate the L2-norm ... and rank them" [cite: 1455]
        scores = norm(U, axis=1)
        return scores

    def fit(self, X: Dict[str, np.ndarray], y=None) -> FitResult:
        """
        Apply RHLC feature selection on each view.
        """
        sorted_keys = sorted(X.keys())
        # Assume all views have same number of samples
        n_samples = list(X.values())[0].shape[0]

        # Heuristic for total features to select (same as other baselines)
        k_total = int(n_samples / np.log(n_samples)) if n_samples > 1 else 1

        selected = {}

        for name in sorted_keys:
            data = X[name]
            # Standardization is crucial for regression-based selection
            data_std = StandardScaler().fit_transform(data)

            # Run RHLC for this view
            scores = self._solve_rhlc_view(data_std)

            # Determine number of features to select for this view
            # (Proportional allocation based on view dimensions)
            n_feats = data.shape[1]
            total_dims = sum(v.shape[1] for v in X.values())
            k_view = max(1, int(k_total * (n_feats / total_dims)))

            # Select top indices
            top_idx = np.argsort(scores)[::-1][:k_view]
            selected[name] = top_idx.tolist()

        return FitResult(selected_features=selected)


class NSGL(BaseMethod):
    """
    Corrected Baseline 3: NSGL (Multi-view feature selection via Nonnegative Structured Graph Learning)
    Reference: Bai et al., Neurocomputing 2020.

    This implementation performs joint Non-negative Matrix Factorization across views
    with L2,1-norm sparsity regularization and Graph Laplacian regularization (Manifold Learning).
    """

    def __init__(self, name="NSGL", n_clusters=3, alpha=1.0, beta=0.1, max_iter=50, tol=1e-4, **kwargs):
        """
        Args:
            n_clusters: Number of clusters (k).
            alpha: Regularization parameter for L2,1 norm (Sparsity).
            beta: Regularization parameter for Graph Laplacian (Manifold smoothness).
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.
        """
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

    def _construct_graph(self, F, k=5):
        """
        Construct a similarity graph (W) based on current representation F.
        Using KNN for robustness as a simplified 'Structured Graph'.
        """
        n_samples = F.shape[0]
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(F)
        distances, indices = nbrs.kneighbors(F)

        # Construct adjacency matrix S
        S = np.zeros((n_samples, n_samples))
        sigma = np.mean(distances) + 1e-8

        for i in range(n_samples):
            # Gaussian kernel based on learned representation
            for j_idx, neighbor_idx in enumerate(indices[i]):
                if i == neighbor_idx: continue
                dist = distances[i][j_idx]
                S[i, neighbor_idx] = np.exp(-dist ** 2 / (2 * sigma ** 2))

        # Symmetrize
        S = (S + S.T) / 2
        return S

    def fit(self, X: dict, y=None) -> FitResult:
        # 1. Data Preprocessing (Non-negative requirement)
        sorted_keys = sorted(X.keys())
        X_mat = []  # List of (n, d_v)

        # Use MinMaxScaler to ensure X >= 0 for NMF-style updates
        scaler = MinMaxScaler()

        for k in sorted_keys:
            X_mat.append(scaler.fit_transform(X[k]))

        n_samples = X_mat[0].shape[0]
        n_views = len(X_mat)
        n_clusters = self.n_clusters

        # 2. Initialization
        # Initialize F using K-Means on concatenated data (Warm Start)
        X_concat = np.hstack(X_mat)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_concat)

        # F: (n, k) - One-hot-like initialization + noise to be strictly positive
        F = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            F[i, labels[i]] = 1.0
        F = F + 0.01  # Avoid zero division

        # W_v: List of (d_v, k)
        W = []
        for v in range(n_views):
            d_v = X_mat[v].shape[1]
            # Random initialization
            W.append(np.random.rand(d_v, n_clusters))

        # 3. Iterative Optimization
        # Objective: min sum_v ||X_v^T W_v - F||^2 + alpha ||W_v||_2,1 + beta Tr(F^T L F)

        for iter_num in range(self.max_iter):
            F_prev = F.copy()

            # --- Update W_v for each view ---
            # Rule: W_ij <- W_ij * ( (X F)_ij / (X X^T W + alpha D W)_ij )
            for v in range(n_views):
                X_v = X_mat[v].T  # (d, n) - Paper typically implies X is (d, n), code inputs are (n, d)
                # Let's verify dimensions:
                # Term: ||X^T W - F||^2. Here X input is (n, d). So X^T is (d, n) is wrong.
                # Standard NMF regression: || X W - F ||^2 where X(n, d), W(d, k), F(n, k).
                # Let's assume standard latent formulation: X_v ~ F W_v^T.
                # BUT feature selection papers usually use: || X_v W_v - F ||^2 to project features to labels.
                # Let's stick to: min || X_v W_v - F ||_F^2

                # Input X[v] is (n, d). W[v] is (d, k). F is (n, k).
                XtX = X_mat[v].T @ X_mat[v]  # (d, d)
                XtF = X_mat[v].T @ F  # (d, k)

                # L2,1 Norm regularization derivative Matrix D
                # D is diagonal with d_ii = 1 / (2 ||w_i||).
                # In Multiplicative Update, this usually appears in the denominator.
                row_norms = np.linalg.norm(W[v], axis=1) + 1e-8
                D_diag = 1.0 / (2 * row_norms)  # (d,)
                # Broadcasting alpha * D * W
                # (d, k)
                Reg_term = self.alpha * (W[v].T * D_diag).T

                # Multiplicative Update for W
                numerator = XtF
                denominator = (XtX @ W[v]) + Reg_term + 1e-8

                W[v] = W[v] * (numerator / denominator)

            # --- Update F ---
            # Term: sum || X W - F ||^2 + beta Tr(F^T L F)
            # Derivative wrt F: 2 sum (F - X W) + 2 beta L F = 0
            # (sum I + beta L) F = sum X W
            # However, for non-negative F, we typically use MUR.
            # Gradient positive part: (sum I) F + beta L_pos F
            # Gradient negative part: sum X W + beta L_neg F
            # Note: L = D - S. L_pos = D, L_neg = S.

            # 1. Update Graph Structure S based on current F (Adaptive Graph)
            S = self._construct_graph(F)
            # Compute Degree Matrix D
            D_vec = np.sum(S, axis=1)
            D_mat = np.diag(D_vec)
            # L = D - S

            # 2. MUR for F
            # Numerator: sum (X_v W_v) + beta * S * F
            # Denominator: n_views * F + beta * D * F

            num_F = np.zeros_like(F)
            for v in range(n_views):
                num_F += X_mat[v] @ W[v]

            num_F += self.beta * (S @ F)

            denom_F = n_views * F + self.beta * (D_mat @ F) + 1e-8

            F = F * (num_F / denom_F)

            # Normalize F (often needed in NMF to prevent scaling issues)
            # F = F / (np.max(F) + 1e-8)

            # Check convergence
            if np.linalg.norm(F - F_prev) / (np.linalg.norm(F_prev) + 1e-8) < self.tol:
                break

        # 4. Feature Selection
        # Score = L2 norm of W rows across all views (or per view)
        # We need to map back to global indices

        selected = {}

        # Heuristic: Select top features per view based on W_v
        # K_total logic from other baselines
        total_samples = X_mat[0].shape[0]
        k_total = int(total_samples / np.log(total_samples)) if total_samples > 1 else 1

        for v, name in enumerate(sorted_keys):
            scores = np.linalg.norm(W[v], axis=1)  # (d_v, )

            # Determine number of features for this view (Proportional)
            n_feats = W[v].shape[0]
            total_dims = sum(w.shape[0] for w in W)
            k_view = max(1, int(k_total * (n_feats / total_dims)))

            top_idx = np.argsort(scores)[::-1][:k_view]
            selected[name] = top_idx.tolist()

        return FitResult(selected_features=selected)