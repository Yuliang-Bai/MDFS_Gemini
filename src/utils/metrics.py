import numpy as np
from typing import Dict, List


def calculate_selection_metrics(true_features: Dict[str, List[int]],
                                selected_features: Dict[str, List[int]]) -> Dict[str, float]:
    """
    同时计算 Recall 和 Precision (包含 Total 和 Per-View)
    """
    metrics = {}

    total_inter = 0
    total_true = 0
    total_sel = 0

    all_views = sorted(true_features.keys())

    for modality in all_views:
        true_idxs = set(true_features[modality])
        pred_idxs = set(selected_features.get(modality, []))

        # Intersection
        inter = len(true_idxs & pred_idxs)
        n_true = len(true_idxs)
        n_sel = len(pred_idxs)

        total_inter += inter
        total_true += n_true
        total_sel += n_sel

        # Per-View Metrics
        metrics[f"recall_{modality}"] = inter / n_true if n_true > 0 else 0.0
        metrics[f"precision_{modality}"] = inter / n_sel if n_sel > 0 else 0.0

    # Total Metrics
    metrics["recall_total"] = total_inter / total_true if total_true > 0 else 0.0
    metrics["precision_total"] = total_inter / total_sel if total_sel > 0 else 0.0

    return metrics


# 保留旧接口以兼容旧代码 (可选)
calculate_recall = calculate_selection_metrics