import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import multiprocessing
import pandas as pd
from src.methods.classification.proposed import MDFSClassifier
from src.methods.classification.baselines import SCFS, HLRFS
from src.simulations.classification import generate_classification_data
from src.utils.metrics import calculate_recall

CONFIG = {
    "n_repeats": 5, "n_cores": 2, "n_samples": 200, "dims": [100, 100], "n_classes": 3,
    "mdfs_params": {"latent_dim": 10}, "base_params": {}
}

def run_single_trial(seed):
    X, y, true_feats = generate_classification_data(CONFIG["n_samples"], CONFIG["dims"], CONFIG["n_classes"], seed=seed)
    res = {}
    m = MDFSClassifier(**CONFIG["mdfs_params"])
    res["MDFS_Recall"] = calculate_recall(true_feats, m.fit(X, y).selected_features)["recall_total"]
    b = SCFS(**CONFIG["base_params"])
    res["SCFS_Recall"] = calculate_recall(true_feats, b.fit(X, y).selected_features)["recall_total"]
    return res

if __name__ == "__main__":
    print("Running Classification Simulation...")
    with multiprocessing.Pool(CONFIG["n_cores"]) as pool:
        results = pool.map(run_single_trial, range(CONFIG["n_repeats"]))
    df = pd.DataFrame(results)
    print(df.describe())
    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/sim_2_cls.csv", index=False)
