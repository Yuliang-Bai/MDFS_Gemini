import numpy as np
from typing import Dict, List, Optional
from ...base import BaseMethod, FitResult

class MDFSRegressor(BaseMethod):
    """MDFS 回归专用版"""
    def __init__(self, name="MDFS_Reg", latent_dim=10, temperature=1.0, hidden_ratio=0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.hidden_ratio = hidden_ratio
        self.epsilon = 1e-8

    def _preprocess(self, X):
        X_std = {}
        for k, v in X.items():
            mean = np.mean(v, axis=0)
            std = np.std(v, axis=0) + self.epsilon
            X_std[k] = (v - mean) / std
        return X_std

    def _pca_encode(self, x, out_dim):
        x_centered = x - x.mean(axis=0)
        u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        d = min(out_dim, vt.shape[0])
        return x_centered @ vt[:d].T

    def _get_gating_weights(self, reps, y):
        scores = []
        for r in reps:
            corrs = [np.abs(np.corrcoef(r[:, j], y)[0, 1]) for j in range(r.shape[1])]
            scores.append(np.mean(corrs))
        scores = np.array(scores)
        logits = scores / self.temperature
        exp_scores = np.exp(logits - np.max(logits))
        return exp_scores / (np.sum(exp_scores) + self.epsilon)

    def _mrdc_screen(self, X, z):
        n = z.shape[0]
        k_total = int(n / np.log(n)) if n > 2 else 1
        selected = {}
        total_dims = sum(v.shape[1] for v in X.values())
        for name, data in X.items():
            k_m = max(1, int(k_total * (data.shape[1] / total_dims)))
            corrs = []
            for j in range(data.shape[1]):
                c = np.mean([np.abs(np.corrcoef(data[:, j], z[:, l])[0,1]) for l in range(z.shape[1])])
                corrs.append(c)
            idx = np.argsort(corrs)[::-1][:k_m]
            selected[name] = idx.tolist()
        return selected

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> FitResult:
        X_std = self._preprocess(X)
        reps = [self._pca_encode(v, int(v.shape[1]*self.hidden_ratio)) for v in X_std.values()]
        alpha = self._get_gating_weights(reps, y)
        z_joint = np.hstack([r * w for r, w in zip(reps, alpha)])
        z_final = self._pca_encode(z_joint, self.latent_dim)
        selected = self._mrdc_screen(X_std, z_final)
        return FitResult(selected_features=selected, model_state={"alpha": alpha})
