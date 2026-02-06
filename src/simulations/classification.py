import numpy as np
def generate_classification_data(n_samples=200, n_features=[300, 300], n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    X = {f"view{i+1}": rng.standard_normal((n_samples, p)) for i, p in enumerate(n_features)}
    y = rng.integers(0, n_classes, size=n_samples)
    true_features = {}
    for i, (name, data) in enumerate(X.items()):
        true_idx = [0, 1, 2, 3, 4]
        true_features[name] = true_idx
        for cls in range(n_classes):
            mask = (y == cls)
            shift = rng.uniform(-1, 1, size=len(true_idx))
            data[np.ix_(mask, true_idx)] += shift
    return X, y, true_features
