import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
from scipy.sparse import csgraph
from scipy.linalg import eigh
from ...base import BaseMethod, FitResult


class MCFL(BaseMethod):
    """
    Baseline 1: MCFL (Multi-view Clustering and Feature Learning)
    Reference: Wang et al., "Multi-view clustering and feature learning via structured sparsity", 2013.
    """

    def __init__(self, name="MCFL", n_clusters=3, gamma=0.1, max_iter=20, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter

    def fit(self, X: dict, y=None) -> FitResult:
        # 1. Concat & Normalize
        sorted_keys = sorted(X.keys())
        X_list = [X[k] for k in sorted_keys]
        X_concat = np.hstack(X_list)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_concat)

        n_samples, n_features = X_std.shape

        # 2. Initialize Weights (Uniform)
        w = np.ones(n_features) / n_features

        # 3. Iterative Optimization
        for t in range(self.max_iter):
            # Step A: Weighted K-Means
            X_weighted = X_std * np.sqrt(w + 1e-8)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=42)
            kmeans.fit(X_weighted)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            # Step B: Update Weights via L2,1 assumption
            scores = np.zeros(n_features)
            for k in range(self.n_clusters):
                cluster_mask = (labels == k)
                if np.sum(cluster_mask) == 0: continue
                diff = X_std[cluster_mask] - centers[k]
                scores += np.sum(diff ** 2, axis=0)

            # W_j update
            scores = np.sqrt(scores + 1e-8)
            new_w = 1.0 / (2 * scores + self.gamma)

            # Normalize w
            w = new_w / np.sum(new_w)

        # 4. Selection
        k_total = int(n_samples / np.log(n_samples)) if n_samples > 1 else 1
        top_idx = np.argsort(w)[::-1][:k_total]

        selected = {}
        curr = 0
        for i, name in enumerate(sorted_keys):
            dim = X[name].shape[1]
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class MRAG(BaseMethod):
    """
    Baseline 2: MRAG (Multi-view Robust Affinity Graph)
    Reference: Jing et al., "Learning robust affinity graph representation for multi-view clustering", 2021.
    """

    def __init__(self, name="MRAG", n_clusters=3, k_neighbors=5, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors

    def fit(self, X: dict, y=None) -> FitResult:
        sorted_keys = sorted(X.keys())
        X_concat = np.hstack([X[k] for k in sorted_keys])
        n_samples, n_features = X_concat.shape

        # 1. Robust Graph Construction (Simplified as average KNN graph)
        W_sum = np.zeros((n_samples, n_samples))
        for key in sorted_keys:
            data = X[key]
            data = StandardScaler().fit_transform(data)
            A = kneighbors_graph(data, n_neighbors=self.k_neighbors, mode='connectivity', include_self=False)
            W_sum += A.toarray()

        W = W_sum / len(sorted_keys)
        W = (W + W.T) / 2

        # 2. Laplacian Matrix
        L = csgraph.laplacian(W, normed=True)

        # 3. Feature Scoring (Laplacian Score)
        scores = []
        X_std = StandardScaler().fit_transform(X_concat)
        for j in range(n_features):
            f = X_std[:, j]
            var = np.var(f)
            if var < 1e-8:
                scores.append(float('inf'))
            else:
                s = f.T @ L @ f
                scores.append(s)

        scores = np.array(scores)

        # Select Smallest Scores
        k_total = int(n_samples / np.log(n_samples)) if n_samples > 1 else 1
        top_idx = np.argsort(scores)[:k_total]

        selected = {}
        curr = 0
        for i, name in enumerate(sorted_keys):
            dim = X[name].shape[1]
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)


class NSGL(BaseMethod):
    """
    Baseline 3: NSGL (Nonnegative Structured Graph Learning)
    Reference: Bai et al., "Multi-view feature selection via Nonnegative Structured Graph Learning", 2020.
    """

    def __init__(self, name="NSGL", n_clusters=3, alpha=0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha

    def fit(self, X: dict, y=None) -> FitResult:
        sorted_keys = sorted(X.keys())
        X_list = []
        dims = []
        for k in sorted_keys:
            m = MinMaxScaler()
            X_list.append(m.fit_transform(X[k]))
            dims.append(X[k].shape[1])

        X_concat = np.hstack(X_list)
        n_samples, n_features = X_concat.shape

        # NMF: X.T ~ W * H
        X_target = X_concat.T

        nmf = NMF(n_components=self.n_clusters, init='nndsvd', random_state=42, max_iter=200)
        W_mat = nmf.fit_transform(X_target)

        # Feature Score: L2 norm of rows of W
        scores = np.linalg.norm(W_mat, axis=1)

        # Select Largest Scores
        k_total = int(n_samples / np.log(n_samples)) if n_samples > 1 else 1
        top_idx = np.argsort(scores)[::-1][:k_total]

        selected = {}
        curr = 0
        for i, name in enumerate(sorted_keys):
            dim = dims[i]
            sel = [idx - curr for idx in top_idx if curr <= idx < curr + dim]
            selected[name] = sel
            curr += dim

        return FitResult(selected_features=selected)