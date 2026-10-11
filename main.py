"""
main.py
-------
Training pipeline for the TruthLens fake news detector.

Run with:
    python main.py                      # download FakeNewsNet (if needed) + train
    python main.py --no-plots --no-cv   # faster run, skips figures and CV
    python main.py --data synthetic     # offline smoke run (demo model only!)

Steps:
1. Acquire dataset (FakeNewsNet headlines, cached under data/raw/)
2. Text preprocessing
3. Stratified train / test split
4. TF-IDF feature extraction
5. Model training (Logistic Regression + linear SVM, class-balanced)
6. Held-out evaluation (overall, per subset, per verdict band)
7. Figures → reports/figures/   (optional)
8. Artefacts + model card → models/
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_fakenewsnet, generate_dataset, load_fakenewsnet  # noqa: E402
from src.evaluate import (  # noqa: E402
    compare_models,
    compute_band_metrics,
    compute_metrics,
    compute_subset_metrics,
    get_classification_report,
    get_roc_data,
    print_evaluation_report,
)
from src.features import TFIDFExtractor  # noqa: E402
from src.models import FakeNewsClassifier  # noqa: E402
from src.predict import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    DEMO_HEADLINES,
    MODEL_CARD_NAME,
    predict_headlines,
)
from src.preprocessor import preprocess_dataframe  # noqa: E402

CONFIG = {
    "test_size":     0.20,
    "random_state":  42,
    "figures_dir":   "reports/figures",
    "models_dir":    "models",
    "data_dir":      "data/raw/fakenewsnet",
    "processed_dir": "data/processed",
    # Regularisation chosen by 5-fold CV ROC-AUC on the training split only.
    "lr_C":          2.0,
    "svm_C":         0.1,
    # The served model. LR is used because its coefficients give per-word
    # explanations; held-out quality is on par with the SVM.
    "served_model":  "Logistic Regression",
}

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression", "logistic_regression.pkl"),
    "SVM (Linear)":        ("svm", "svm_linear.pkl"),
}


def banner(title: str):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


def load_data(source: str, synthetic_n: int):
    if source == "synthetic":
        print("[Main] ⚠ Using SYNTHETIC template data — the resulting model is a demo "
              "and will be flagged as such in the app.")
        df = generate_dataset(n_samples=synthetic_n)
        df["subset"] = "synthetic"
        return df, {
            "name": "synthetic-templates",
            "description": "Template-generated headlines (offline demo only)",
        }

    checksums = download_fakenewsnet(CONFIG["data_dir"])
    df = load_fakenewsnet(CONFIG["data_dir"])
    return df, {
        "name": "FakeNewsNet (PolitiFact + GossipCop headlines)",
        "source": "https://github.com/KaiDMML/FakeNewsNet",
        "citation": "Shu, Mahudeswaran, Wang, Lee, Liu. FakeNewsNet: A Data Repository "
                    "with News Content, Social Context and Spatialtemporal Information "
                    "for Studying Fake News on Social Media. arXiv:1809.01286 (2018).",
        "sha256": checksums,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the TruthLens fake news detector")
    parser.add_argument("--data", choices=["fakenewsnet", "synthetic"], default="fakenewsnet")
    parser.add_argument("--synthetic-n", type=int, default=6000)
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--models-dir", default=CONFIG["models_dir"])
    args = parser.parse_args(argv)
    if args.data == "synthetic":
        args.no_plots = True  # never overwrite the published reports with demo-model results

    warnings.filterwarnings("ignore", category=FutureWarning)
    t_start = time.time()
    models_dir = args.models_dir
    for d in [CONFIG["figures_dir"], models_dir, CONFIG["processed_dir"]]:
        os.makedirs(d, exist_ok=True)

    if not args.no_plots:
        from src import visualize as viz

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    banner("STEP 1 — Dataset")
    df_raw, dataset_info = load_data(args.data, args.synthetic_n)

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    banner("STEP 2 — Text Preprocessing")
    df = preprocess_dataframe(df_raw, text_col="text")
    df.to_csv(os.path.join(CONFIG["processed_dir"], f"{args.data}_preprocessed.csv"), index=False)
    if not args.no_plots:
        viz.plot_class_distribution(
            df["label"].values, os.path.join(CONFIG["figures_dir"], "class_distribution.png"))

    # ── 3. Split (stratified on label × subset) ──────────────────────────────
    banner("STEP 3 — Train / Test Split")
    strata = df["label"].astype(str) + "_" + df["subset"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=strata)
    y_train, y_test = train_df["label"].values, test_df["label"].values
    print(f"[Main] Train: {len(train_df):,}  |  Test: {len(test_df):,}  |  "
          f"fake share: {y_train.mean():.3f}")

    # ── 4. TF-IDF ────────────────────────────────────────────────────────────
    banner("STEP 4 — TF-IDF Feature Extraction")
    extractor = TFIDFExtractor(max_features=50000, ngram_range=(1, 2),
                               min_df=2, max_df=0.90, sublinear_tf=True)
    X_train = extractor.fit_transform(train_df["cleaned_text"].tolist())
    X_test = extractor.transform(test_df["cleaned_text"].tolist())
    extractor.save(os.path.join(models_dir, "tfidf_vectorizer.pkl"))

    # ── 5. Training ──────────────────────────────────────────────────────────
    banner("STEP 5 — Model Training")
    models = {
        "Logistic Regression": FakeNewsClassifier("logistic_regression", C=CONFIG["lr_C"]),
        "SVM (Linear)":        FakeNewsClassifier("svm", C=CONFIG["svm_C"]),
    }
    cv_results = {}
    for name, clf in models.items():
        cv_results[name] = clf.train(X_train, y_train, cv=not args.no_cv)
        clf.save(os.path.join(models_dir, MODEL_FILES[name][1]))

    # ── 6. Evaluation ────────────────────────────────────────────────────────
    banner("STEP 6 — Held-out Evaluation")
    all_metrics, roc_data, details = {}, {}, {}
    low, high = DEFAULT_THRESHOLDS["low"], DEFAULT_THRESHOLDS["high"]
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_risk_score(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics[name] = metrics
        roc_data[name] = get_roc_data(y_test, y_prob)
        details[name] = {
            "subsets": compute_subset_metrics(y_test, y_prob, test_df["subset"].values),
            "bands": compute_band_metrics(y_test, y_prob, low, high),
        }
        print_evaluation_report(name, metrics)
        print(get_classification_report(y_test, y_pred))
        for subset, m in details[name]["subsets"].items():
            print(f"  {subset:<11} n={m['n']:<5} auc={m.get('roc_auc', float('nan')):.3f} "
                  f"acc={m['accuracy']:.3f} f1_fake={m['f1_fake']:.3f}")
        for band, m in details[name]["bands"].items():
            print(f"  band {band:<12} share={m['share']:.3f} actually_fake={m['actually_fake']}")

        if not args.no_plots:
            short = name.split()[0].lower()
            viz.plot_confusion_matrix(
                metrics["confusion_matrix"], name,
                os.path.join(CONFIG["figures_dir"], f"confusion_matrix_{short}.png"))
            viz.plot_risk_score_distribution(
                y_test, y_prob, name,
                os.path.join(CONFIG["figures_dir"], f"risk_distribution_{short}.png"))

    comparison_df = compare_models(all_metrics)
    print("\n", comparison_df.to_string(), "\n")
    if args.data != "synthetic":
        comparison_df.to_csv(os.path.join("reports", "model_comparison.csv"))

    # ── 7. Figures ───────────────────────────────────────────────────────────
    lr_model = models["Logistic Regression"].model
    coef = lr_model.coef_.ravel()
    names = np.asarray(extractor.get_feature_names())
    top_fake_idx = np.argsort(coef)[::-1][:20]
    top_real_idx = np.argsort(coef)[:20]
    print("  Top FAKE terms:", ", ".join(names[top_fake_idx[:10]]))
    print("  Top REAL terms:", ", ".join(names[top_real_idx[:10]]))

    if not args.no_plots:
        banner("STEP 7 — Figures")
        viz.plot_roc_curves(roc_data, os.path.join(CONFIG["figures_dir"], "roc_curve_comparison.png"))
        viz.plot_model_comparison(
            comparison_df, os.path.join(CONFIG["figures_dir"], "model_comparison.png"))
        viz.plot_feature_importance(
            names[top_real_idx].tolist(), names[top_fake_idx].tolist(),
            os.path.join(CONFIG["figures_dir"], "feature_importance.png"),
            real_weights=coef[top_real_idx].tolist(), fake_weights=coef[top_fake_idx].tolist(),
            xlabel="Logistic regression |coefficient|")
        sizes, tr_scores, val_scores = learning_curve(
            FakeNewsClassifier("logistic_regression", C=CONFIG["lr_C"]).model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=StratifiedKFold(5, shuffle=True, random_state=CONFIG["random_state"]),
            scoring="roc_auc", n_jobs=-1)
        viz.plot_learning_curve(sizes, tr_scores, val_scores, "Logistic Regression",
                                os.path.join(CONFIG["figures_dir"], "learning_curve_lr.png"))

    # ── 8. Model card ────────────────────────────────────────────────────────
    banner("STEP 8 — Model Card")
    import sklearn

    served = CONFIG["served_model"]
    model_type, model_file = MODEL_FILES[served]
    m = all_metrics[served]
    card = {
        "model_name": served,
        "model_type": model_type,
        "model_file": model_file,
        "vectorizer_file": "tfidf_vectorizer.pkl",
        "is_demo": args.data == "synthetic",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "thresholds": {"low": low, "high": high},
        "dataset": {
            **dataset_info,
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "fake_share": round(float(df["label"].mean()), 4),
        },
        "metrics": {
            "test": {k: round(float(m[k]), 4) for k in
                     ("accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision")},
            "cv_roc_auc": ({k: round(float(v), 4) for k, v in cv_results[served].items()}
                           if cv_results[served] else None),
            **details[served],
            "comparison": {
                name: {"roc_auc": round(float(mm["roc_auc"]), 4), "f1": round(float(mm["f1"]), 4)}
                for name, mm in all_metrics.items()
            },
        },
    }
    card_path = os.path.join(models_dir, MODEL_CARD_NAME)
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(f"[Main] Model card → {card_path}")

    # ── Demo ─────────────────────────────────────────────────────────────────
    banner("Demo Predictions (served model)")
    predict_headlines(DEMO_HEADLINES, extractor, models[served], verbose=True)

    banner("PIPELINE COMPLETE ✅")
    print(f"  Time: {time.time() - t_start:.1f}s  |  served: {served}  |  "
          f"test ROC-AUC {m['roc_auc']:.4f}  |  F1(fake) {m['f1']:.4f}\n")
    return card


if __name__ == "__main__":
    main()
