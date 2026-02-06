import numpy as np
def generate_regression_data(n_samples=200, n_features=[300, 300], noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = {f"view{i+1}": rng.standard_normal((n_samples, p)) for i, p in enumerate(n_features)}
    y = np.zeros(n_samples)
    true_features = {}
    for i, (name, data) in enumerate(X.items()):
        true_idx = [0, 1, 2, 3, 4]
        true_features[name] = true_idx
        w = rng.uniform(1, 2, size=len(true_idx))
        y += data[:, true_idx] @ w
    y += noise * rng.standard_normal(n_samples)
    return X, y, true_features
