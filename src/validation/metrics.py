from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    Compute ROC-AUC with basic safety checks.
    """
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(roc_auc_score(y_true_arr, y_pred_arr))


def fold_metrics_summary(fold_scores: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(fold_scores), dtype=float)
    return {
        "mean_auc": float(arr.mean()),
        "std_auc": float(arr.std()),
        "min_auc": float(arr.min()),
        "max_auc": float(arr.max()),
    }

