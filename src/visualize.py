"""
visualize.py
------------
All visualization utilities for the Fake News Detection project.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Tuple

# ── Style configuration ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

REAL_COLOR = "#2196F3"   # Blue  — real news
FAKE_COLOR = "#F44336"   # Red   — fake news
NEUTRAL_COLOR = "#9C27B0"


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")


# ── 1. Class distribution ────────────────────────────────────────────────────

def plot_class_distribution(y, save_path: str):
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["Real News", "Fake News"]
    colors = [REAL_COLOR, FAKE_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=16, fontweight="bold", y=1.02)

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    axes[0].set_title("Sample Count per Class")
    axes[0].set_ylabel("Number of Samples")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     f"{count:,}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylim(0, max(counts) * 1.15)

    # Pie chart
    axes[1].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 12})
    axes[1].set_title("Class Balance")

    plt.tight_layout()
    _save(fig, save_path)


# ── 2. Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str, save_path: str):
    labels = ["Real News", "Fake News"]
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for annotation
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"shrink": 0.8})

    # Custom annotations: count + percentage
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j + 0.5, i + 0.4, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=16,
                    fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.65, f"({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Risk annotations
    ax.text(0.5, -0.12, "↑ Censorship Risk (FP)", transform=ax.transAxes,
            ha="center", fontsize=9, color=FAKE_COLOR, style="italic")
    ax.text(-0.18, 0.25, "← Misinfo Risk (FN)", transform=ax.transAxes,
            ha="center", fontsize=9, color=REAL_COLOR, style="italic", rotation=90)

    plt.tight_layout()
    _save(fig, save_path)


# ── 3. ROC curves ────────────────────────────────────────────────────────────

def plot_roc_curves(roc_data: Dict[str, Tuple], save_path: str):
    """
    roc_data: {model_name: (fpr, tpr, auc)}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [REAL_COLOR, FAKE_COLOR, NEUTRAL_COLOR]

    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {auc:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (Censorship Risk)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Detection Rate)", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate optimal region
    ax.annotate("Optimal\nRegion", xy=(0.1, 0.9), fontsize=10,
                color="green", style="italic",
                arrowprops=dict(arrowstyle="->", color="green"),
                xytext=(0.2, 0.75))

    plt.tight_layout()
    _save(fig, save_path)


# ── 4. Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(top_real: List[str], top_fake: List[str],
                             save_path: str, n_show: int = 15):
    top_real = top_real[:n_show]
    top_fake = top_fake[:n_show]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Top TF-IDF Features by Class", fontsize=16, fontweight="bold")

    # Real news features
    y_pos = range(len(top_real))
    axes[0].barh(y_pos, range(len(top_real), 0, -1),
                 color=REAL_COLOR, alpha=0.8, edgecolor="white")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_real, fontsize=10)
    axes[0].set_title("Top Features → Real News", color=REAL_COLOR, fontweight="bold")
    axes[0].set_xlabel("TF-IDF Rank (higher = more important)")
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()

    # Fake news features
    axes[1].barh(y_pos, range(len(top_fake), 0, -1),
                 color=FAKE_COLOR, alpha=0.8, edgecolor="white")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_fake, fontsize=10)
    axes[1].set_title("Top Features → Fake News", color=FAKE_COLOR, fontweight="bold")
    axes[1].set_xlabel("TF-IDF Rank (higher = more important)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    _save(fig, save_path)


# ── 5. Model comparison bar chart ────────────────────────────────────────────

def plot_model_comparison(comparison_df, save_path: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available = [m for m in metrics if m in comparison_df.columns]
    df = comparison_df[available]

    n_models = len(df)
    n_metrics = len(available)
    x = np.arange(n_metrics)
    width = 0.35
    colors_list = [REAL_COLOR, FAKE_COLOR, NEUTRAL_COLOR]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_name, row) in enumerate(df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, row[available].values, width * 0.9,
                      label=model_name, color=colors_list[i % len(colors_list)],
                      alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ── 6. Risk score distribution ───────────────────────────────────────────────

def plot_risk_score_distribution(y_true, risk_scores: np.ndarray,
                                  model_name: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Risk Score (Fake News Probability) Distribution — {model_name}",
                 fontsize=14, fontweight="bold")

    real_scores = risk_scores[y_true == 0]
    fake_scores = risk_scores[y_true == 1]

    # KDE plot
    axes[0].hist(real_scores, bins=50, color=REAL_COLOR, alpha=0.6,
                 label="Real News", density=True, edgecolor="white")
    axes[0].hist(fake_scores, bins=50, color=FAKE_COLOR, alpha=0.6,
                 label="Fake News", density=True, edgecolor="white")
    axes[0].axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision Threshold (0.5)")
    axes[0].set_xlabel("Risk Score P(Fake)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Class")
    axes[0].legend()

    # Box plot
    data_to_plot = [real_scores, fake_scores]
    bp = axes[1].boxplot(data_to_plot, patch_artist=True,
                          labels=["Real News", "Fake News"],
                          medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor(REAL_COLOR)
    bp["boxes"][1].set_facecolor(FAKE_COLOR)
    for box in bp["boxes"]:
        box.set_alpha(0.8)
    axes[1].set_ylabel("Risk Score P(Fake)")
    axes[1].set_title("Score Spread by Class")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ── 7. Learning curve ────────────────────────────────────────────────────────

def plot_learning_curve(train_sizes, train_scores, val_scores,
                         model_name: str, save_path: str):
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_mean, "o-", color=REAL_COLOR,
            lw=2, markersize=6, label="Training Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=REAL_COLOR)
    ax.plot(train_sizes, val_mean, "s-", color=FAKE_COLOR,
            lw=2, markersize=6, label="Validation Score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=FAKE_COLOR)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Learning Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()
    _save(fig, save_path)
