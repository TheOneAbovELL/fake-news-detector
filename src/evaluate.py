"""
evaluate.py
-----------
Model evaluation: metrics, confusion matrix, classification reports.
Frames evaluation as a risk modeling problem.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, average_precision_score
)
from typing import Dict, Any


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
    print(f"\n  ── Confusion Matrix ──")
    print(f"     TP (fake caught)    : {metrics['true_positives']}")
    print(f"     FP (real flagged)   : {metrics['false_positives']}  ← censorship risk")
    print(f"     TN (real passed)    : {metrics['true_negatives']}")
    print(f"     FN (fake missed)    : {metrics['false_negatives']}  ← misinformation risk")
    print(f"\n  ── Risk Scores ──")
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
