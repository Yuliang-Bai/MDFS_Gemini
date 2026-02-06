import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

def calculate_recall(true_features: Dict[str, List[int]], 
                     selected_features: Dict[str, List[int]]) -> Dict[str, float]:
    """计算特征筛选查全率 (Recall)"""
    metrics = {}
    total_inter = 0
    total_true = 0

    for modality, true_idxs in true_features.items():
        true_set = set(true_idxs)
        pred_set = set(selected_features.get(modality, []))

        if len(true_set) == 0:
            metrics[f"recall_{modality}"] = 1.0 if len(pred_set) == 0 else 0.0
            continue

        intersection = len(true_set & pred_set)
        metrics[f"recall_{modality}"] = intersection / len(true_set)

        total_inter += intersection
        total_true += len(true_set)

    metrics["recall_total"] = total_inter / total_true if total_true > 0 else 0.0
    return metrics

def evaluate_regression(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def evaluate_classification(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def evaluate_clustering(y_true, y_pred):
    return {
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred)
    }
