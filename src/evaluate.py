"""
evaluate.py
-----------
Model evaluation: metrics, confusion matrix, classification reports.
Frames evaluation as a risk modeling problem.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    """
    Compute full evaluation metrics.

    Parameters
    ----------
    y_true : array-like  — ground truth labels
    y_pred : array-like  — predicted labels
    y_prob : array-like, optional — predicted probabilities for class 1 (fake)

    Returns
    -------
    dict with all metrics
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)

    # Risk framing
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["true_positives"]  = int(tp)   # Correctly flagged fake news
    metrics["false_positives"] = int(fp)   # Real news wrongly flagged (censorship risk)
    metrics["true_negatives"]  = int(tn)   # Correctly passed real news
    metrics["false_negatives"] = int(fn)   # Missed fake news (misinformation risk)
    metrics["confusion_matrix"] = cm

    # Risk scores
    n = len(y_true)
    metrics["misinformation_risk"] = fn / n   # P(undetected fake)
    metrics["censorship_risk"] = fp / n       # P(wrongly blocked real)

    return metrics


def print_evaluation_report(model_name: str, metrics: Dict[str, Any]):
    """Pretty-print evaluation results."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION REPORT — {model_name.upper()}")
    print(sep)
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score    : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print("\n  ── Confusion Matrix ──")
    print(f"     TP (fake caught)    : {metrics['true_positives']}")
    print(f"     FP (real flagged)   : {metrics['false_positives']}  ← censorship risk")
    print(f"     TN (real passed)    : {metrics['true_negatives']}")
    print(f"     FN (fake missed)    : {metrics['false_negatives']}  ← misinformation risk")
    print("\n  ── Risk Scores ──")
    print(f"     Misinformation Risk : {metrics['misinformation_risk']:.4f}")
    print(f"     Censorship Risk     : {metrics['censorship_risk']:.4f}")
    print(sep)


def get_roc_data(y_true, y_prob):
    """Return FPR, TPR arrays for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return fpr, tpr, auc


def get_classification_report(y_true, y_pred) -> str:
    """Return sklearn classification report string."""
    return classification_report(
        y_true, y_pred,
        target_names=["Real News", "Fake News"],
        digits=4
    )


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame for multiple models.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":        name,
            "Accuracy":     round(m["accuracy"], 4),
            "Precision":    round(m["precision"], 4),
            "Recall":       round(m["recall"], 4),
            "F1-Score":     round(m["f1"], 4),
            "ROC-AUC":      round(m.get("roc_auc", float("nan")), 4),
            "Misinfo Risk": round(m["misinformation_risk"], 4),
            "Censor Risk":  round(m["censorship_risk"], 4),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df


def compute_subset_metrics(y_true, y_prob, subsets, threshold: float = 0.5) -> Dict[str, Dict]:
    """
    Per-subset metrics (e.g. PolitiFact vs GossipCop) — an aggregate score can
    hide that a model is much weaker on one kind of news.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    subsets = np.asarray(subsets)
    out = {}
    for name in sorted(set(subsets.tolist())):
        m = subsets == name
        yt, yp = y_true[m], y_prob[m]
        y_hat = (yp >= threshold).astype(int)
        row = {
            "n": int(m.sum()),
            "fake_share": round(float(yt.mean()), 4),
            "accuracy": round(float(accuracy_score(yt, y_hat)), 4),
            "f1_fake": round(float(f1_score(yt, y_hat, zero_division=0)), 4),
        }
        if len(set(yt.tolist())) == 2:
            row["roc_auc"] = round(float(roc_auc_score(yt, yp)), 4)
        out[str(name)] = row
    return out


def compute_band_metrics(y_true, y_prob, low: float, high: float) -> Dict[str, Dict]:
    """
    How the served three-way verdict behaves on held-out data.

    For each band (likely real / uncertain / likely fake) reports the share of
    texts that land there and the share of those that are actually fake — i.e.
    how often a "LIKELY FAKE" verdict is right.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bands = {
        "likely_real": y_prob < low,
        "uncertain": (y_prob >= low) & (y_prob < high),
        "likely_fake": y_prob >= high,
    }
    out = {}
    for name, m in bands.items():
        n = int(m.sum())
        out[name] = {
            "share": round(n / len(y_true), 4) if len(y_true) else 0.0,
            "n": n,
            "actually_fake": round(float(y_true[m].mean()), 4) if n else None,
        }
    return out
