"""Evaluation metrics geared to fraud: ranking quality, precision at review capacity, thresholds."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    """Precision if the review team can only look at the top k_frac of transactions."""
    k = max(1, int(len(scores) * k_frac))
    top = np.argsort(-scores)[:k]
    return float(np.asarray(y_true)[top].mean())


def evaluate(y_true, scores) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision_at_1pct": precision_at_k(y_true, scores, 0.01),
        "precision_at_5pct": precision_at_k(y_true, scores, 0.05),
        "best_f1": float(f1[best]),
        "best_f1_threshold": float(thresholds[best]) if len(thresholds) else 0.5,
        "base_rate": float(y_true.mean()),
        "n_valid": int(len(y_true)),
    }


def pr_curve_points(y_true, scores, max_points: int = 200) -> dict:
    """Down-sampled PR curve for plotting in the dashboard."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    idx = np.linspace(0, len(thresholds) - 1, min(max_points, len(thresholds))).astype(int)
    return {
        "precision": precision[idx].round(4).tolist(),
        "recall": recall[idx].round(4).tolist(),
        "threshold": thresholds[idx].round(4).tolist(),
    }
