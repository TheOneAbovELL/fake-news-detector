"""
main.py
-------
🚀 Single entry point for the Fake News Detection Pipeline.

Run with:
    python main.py

This script executes the full pipeline end-to-end:
1. Dataset generation
2. Text preprocessing
3. TF-IDF feature extraction
4. Model training (Logistic Regression + SVM)
5. Evaluation with full metrics
6. Visualization generation
7. Model saving
8. Demo predictions
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import generate_dataset
from src.preprocessor  import preprocess_dataframe
from src.features      import TFIDFExtractor
from src.models        import FakeNewsClassifier
from src.evaluate      import (
    compute_metrics, print_evaluation_report,
    get_roc_data, get_classification_report, compare_models
)
from src.visualize     import (
    plot_class_distribution, plot_confusion_matrix, plot_roc_curves,
    plot_feature_importance, plot_model_comparison,
    plot_risk_score_distribution, plot_learning_curve
)
from src.predict       import predict_headlines, DEMO_HEADLINES

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "n_samples":      6000,
    "test_size":      0.20,
    "random_state":   42,
    "figures_dir":    "reports/figures",
    "models_dir":     "models",
    "data_dir":       "data/raw",
    "processed_dir":  "data/processed",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(title: str):
    width = 62
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def section(text: str):
    print(f"\n── {text} " + "─" * max(0, 55 - len(text)))


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("\n╔═════════════════════════════════════╗")
    print("║     FAKE NEWS DETECTION PIPELINE      ║")
    print("╚═══════════════════════════════════════╝")

    # Create output directories
    for d in [CONFIG["figures_dir"], CONFIG["models_dir"],
              CONFIG["data_dir"], CONFIG["processed_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    banner("STEP 1 — Dataset Generation")
    raw_path = os.path.join(CONFIG["data_dir"], "news_headlines.csv")
    df_raw = generate_dataset(
        n_samples=CONFIG["n_samples"],
        save_path=raw_path
    )

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    banner("STEP 2 — Text Preprocessing")
    df = preprocess_dataframe(df_raw, text_col="text")
    processed_path = os.path.join(CONFIG["processed_dir"], "news_preprocessed.csv")
    df.to_csv(processed_path, index=False)
    print(f"[Main] Preprocessed data saved → {processed_path}")

    # Class distribution plot
    section("Plotting class distribution")
    plot_class_distribution(
        df["label"].values,
        os.path.join(CONFIG["figures_dir"], "class_distribution.png")
    )

    # ── Step 3: Train / Test Split ───────────────────────────────────────────
    banner("STEP 3 — Train / Test Split")
    X_text = df["cleaned_text"].values
    y      = df["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y
    )
    print(f"[Main] Train: {len(X_train_text):,}  |  Test: {len(X_test_text):,}")

    # ── Step 4: TF-IDF Feature Extraction ───────────────────────────────────
    banner("STEP 4 — TF-IDF Feature Extraction")
    extractor = TFIDFExtractor(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )
    X_train = extractor.fit_transform(X_train_text.tolist())
    X_test  = extractor.transform(X_test_text.tolist())

    # Save vectorizer
    vectorizer_path = os.path.join(CONFIG["models_dir"], "tfidf_vectorizer.pkl")
    extractor.save(vectorizer_path)

    # Top features
    section("Extracting top TF-IDF features per class")
    top_real, top_fake = extractor.get_top_features_per_class(X_train, y_train, n_top=20)
    print(f"\n  Top REAL features: {', '.join(top_real[:8])}")
    print(f"  Top FAKE features: {', '.join(top_fake[:8])}")

    plot_feature_importance(
        top_real, top_fake,
        os.path.join(CONFIG["figures_dir"], "feature_importance.png")
    )

    # ── Step 5: Model Training ───────────────────────────────────────────────
    banner("STEP 5 — Model Training")
    models = {
        "Logistic Regression": FakeNewsClassifier("logistic_regression", C=1.0),
        "SVM (Linear)":        FakeNewsClassifier("svm", C=1.0),
    }

    trained_models = {}
    for name, clf in models.items():
        section(f"Training {name}")
        clf.train(X_train, y_train, cv=True)
        save_path = os.path.join(CONFIG["models_dir"],
                                  name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl")
        clf.save(save_path)
        trained_models[name] = clf

    # ── Step 6: Evaluation ───────────────────────────────────────────────────
    banner("STEP 6 — Model Evaluation")
    all_metrics = {}
    roc_data    = {}

    for name, clf in trained_models.items():
        y_pred      = clf.predict(X_test)
        y_prob      = clf.predict_risk_score(X_test)
        metrics     = compute_metrics(y_test, y_pred, y_prob)
        all_metrics[name] = metrics

        print_evaluation_report(name, metrics)
        print(f"\n  Classification Report:\n")
        print(get_classification_report(y_test, y_pred))

        # Confusion matrix
        short_name = name.split()[0]
        plot_confusion_matrix(
            metrics["confusion_matrix"], name,
            os.path.join(CONFIG["figures_dir"], f"confusion_matrix_{short_name.lower()}.png")
        )

        # Risk distribution
        plot_risk_score_distribution(
            y_test, y_prob, name,
            os.path.join(CONFIG["figures_dir"], f"risk_distribution_{short_name.lower()}.png")
        )

        # ROC data
        fpr, tpr, auc = get_roc_data(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

    # ROC curve comparison
    section("Plotting ROC curves")
    plot_roc_curves(roc_data, os.path.join(CONFIG["figures_dir"], "roc_curve_comparison.png"))

    # Model comparison
    section("Model comparison")
    comparison_df = compare_models(all_metrics)
    print("\n", comparison_df.to_string(), "\n")
    comparison_df.to_csv(os.path.join("reports", "model_comparison.csv"))

    plot_model_comparison(
        comparison_df,
        os.path.join(CONFIG["figures_dir"], "model_comparison.png")
    )

    # ── Step 7: Learning Curves ──────────────────────────────────────────────
    banner("STEP 7 — Learning Curves")
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression as LR

    # Use a fresh LR pipeline for learning curve (faster)
    lc_clf = LR(C=1.0, max_iter=500, solver="lbfgs", n_jobs=-1, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)
    print("[Main] Computing learning curve (this may take a moment)...")
    sizes, tr_scores, val_scores = learning_curve(
        lc_clf, X_train, y_train,
        train_sizes=train_sizes, cv=5,
        scoring="accuracy", n_jobs=-1
    )
    plot_learning_curve(
        sizes, tr_scores, val_scores, "Logistic Regression",
        os.path.join(CONFIG["figures_dir"], "learning_curve_lr.png")
    )

    # ── Step 8: Demo Predictions ─────────────────────────────────────────────
    banner("STEP 8 — Demo Predictions")
    print("\n  Running predictions on sample headlines...\n")

    best_model = trained_models["SVM (Linear)"]
    results = predict_headlines(DEMO_HEADLINES, extractor, best_model, verbose=True)

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              PREDICTION SUMMARY                      │")
    print("  ├───────────────┬──────────────┬────────────────────── │")
    print("  │ Prediction    │  Risk Score  │  Risk Level           │")
    print("  ├───────────────┼──────────────┼────────────────────── │")
    for r in results:
        icon = "🔴 FAKE" if r["prediction"] == "FAKE" else "🟢 REAL"
        print(f"  │ {icon}       │    {r['risk_score']:.3f}     │  {r['risk_level']:<20} │")
    print("  └─────────────────────────────────────────────────────┘")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner("PIPELINE COMPLETE ✅")
    print(f"\n  ⏱  Total time : {elapsed:.1f} seconds")
    print(f"  📊  Figures   : {CONFIG['figures_dir']}/")
    print(f"  🤖  Models    : {CONFIG['models_dir']}/")
    print(f"  📁  Data      : {CONFIG['data_dir']}/")
    print(f"\n  Best Model    : SVM (Linear)  —  AUC {roc_data['SVM (Linear)'][2]:.4f}")
    print()


if __name__ == "__main__":
    main()
