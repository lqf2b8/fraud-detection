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
        "n_valid": len(y_true),
    }


def threshold_table(y_true, scores, amounts, n_points: int = 120) -> dict:
    """Outcomes at a grid of review budgets (share of transactions flagged, 0.1%–50%).

    For each budget: score threshold, frauds caught, precision, recall, and fraud dollars caught.
    The dashboard combines this with a review cost to find the cheapest operating point.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    amt = np.nan_to_num(np.asarray(amounts, dtype=float))
    order = np.argsort(-s, kind="stable")
    y_sorted, amt_fraud = y[order], (y * amt)[order]
    cum_tp, cum_amt = np.cumsum(y_sorted), np.cumsum(amt_fraud)
    n, total_fraud, total_amt = len(y), y.sum(), (y * amt).sum()

    shares = np.unique(np.geomspace(0.001, 0.5, n_points))
    k = np.unique(np.clip((shares * n).astype(int), 1, n))
    tp = cum_tp[k - 1]
    return {
        "threshold": s[order][k - 1].round(6).tolist(),
        "flag_rate": (k / n).round(6).tolist(),
        "n_flagged": k.tolist(),
        "frauds_caught": tp.astype(int).tolist(),
        "precision": (tp / k).round(4).tolist(),
        "recall": (tp / max(total_fraud, 1)).round(4).tolist(),
        "fraud_amt_caught": cum_amt[k - 1].round(2).tolist(),
        "n_total": int(n),
        "n_fraud": int(total_fraud),
        "fraud_amt_total": round(float(total_amt), 2),
    }


def operating_cost(table: dict, review_cost: float) -> np.ndarray:
    """Cost at each grid point = review cost × flagged + fraud $ missed."""
    flagged = np.asarray(table["n_flagged"], dtype=float)
    missed = table["fraud_amt_total"] - np.asarray(table["fraud_amt_caught"], dtype=float)
    return review_cost * flagged + missed


def cost_optimal_point(table: dict, review_cost: float) -> dict:
    cost = operating_cost(table, review_cost)
    i = int(np.argmin(cost))
    return {
        "review_cost_usd": review_cost,
        "threshold": table["threshold"][i],
        "flag_rate": table["flag_rate"][i],
        "precision": table["precision"][i],
        "recall": table["recall"][i],
        "fraud_amt_caught_share": round(table["fraud_amt_caught"][i] / max(table["fraud_amt_total"], 1), 4),
        "cost_usd": round(float(cost[i]), 2),
        "cost_no_model_usd": round(float(table["fraud_amt_total"]), 2),
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
