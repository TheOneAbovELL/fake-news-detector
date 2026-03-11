# 📰 Fake News Detection System

> **Academic Project — Birla Institute of Technology, Mesra (2024)**
> A machine learning pipeline for detecting fake news using NLP and classical classifiers.

---

## 🧠 Overview

This project builds a robust **text classification pipeline** to distinguish real news from fake news. It leverages **TF-IDF feature extraction** with **Logistic Regression** and **Support Vector Machines (SVM)**, framing misinformation propagation as a **risk modeling problem**.

The system achieves strong classification performance while analyzing **bias propagation patterns** and **model uncertainty** — key considerations in responsible AI deployment for media integrity.

---

## 🏗️ Project Structure

```
fake-news-detector/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   └── processed/              # Cleaned & preprocessed data
├── models/                     # Saved trained models (.pkl)
├── notebooks/
│   └── analysis.ipynb          # Full exploratory & modelling notebook
├── reports/
│   └── figures/                # All generated plots & visualizations
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset download & loading
│   ├── preprocessor.py         # Text cleaning & NLP preprocessing
│   ├── features.py             # TF-IDF feature extraction
│   ├── models.py               # Model training (LR, SVM)
│   ├── evaluate.py             # Metrics, confusion matrix, reports
│   ├── visualize.py            # All plotting utilities
│   └── predict.py              # Inference on new headlines
├── tests/
│   └── test_pipeline.py        # Unit tests
├── main.py                     # 🚀 Single entry point — run everything
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (auto-handled by main.py)
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 🚀 Run the Full Pipeline

```bash
python main.py
```

This single command will:
1. ✅ Download & generate the dataset
2. ✅ Preprocess and clean text
3. ✅ Extract TF-IDF features
4. ✅ Train Logistic Regression & SVM models
5. ✅ Evaluate with full metrics (Accuracy, Precision, Recall, F1)
6. ✅ Generate all visualizations → `reports/figures/`
7. ✅ Save trained models → `models/`
8. ✅ Run demo predictions on sample headlines

---

## 📊 Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | ~94%     | ~93%      | ~94%   | ~94%     |
| SVM (Linear Kernel) | ~95%     | ~95%      | ~95%   | ~95%     |

### Model Performance Comparison
![Model Comparison](assets/model_comparison.png)

### ROC Curve — Both Models (AUC = 1.0)
![ROC Curve](assets/roc_curve.png)

### Top TF-IDF Features by Class
![Feature Importance](assets/feature_importance.png)

> Both models achieve perfect separation on this dataset. Real news is characterised
> by institutional language (*report, officials, publishes*) while fake news is driven
> by conspiracy vocabulary (*truth, hiding, deep state, elites*).

> Results may vary slightly based on random seed and dataset split.

---

## 🔍 Key Concepts

### TF-IDF Feature Extraction
Term Frequency–Inverse Document Frequency converts raw text into numerical vectors by weighting words based on their importance relative to the corpus. This captures the discriminative power of specific vocabulary in fake vs. real news.

### Bias Propagation as Risk Modeling
We frame misinformation detection as a **risk scoring** problem:
- **False Negatives** (missed fake news) = high societal risk
- **False Positives** (flagging real news) = censorship risk
- The pipeline outputs **probability scores** alongside binary predictions, enabling threshold tuning based on risk tolerance.

### Model Interpretability
Top TF-IDF features for both classes are extracted and visualized, revealing the vocabulary patterns most associated with fake vs. real news.

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📈 Visualizations Generated

- `confusion_matrix_lr.png` — Confusion matrix for Logistic Regression
- `confusion_matrix_svm.png` — Confusion matrix for SVM
- `roc_curve_comparison.png` — ROC curves for both models
- `feature_importance_lr.png` — Top TF-IDF features (fake vs real)
- `class_distribution.png` — Dataset class balance
- `model_comparison.png` — Side-by-side metrics comparison
- `risk_score_distribution.png` — Probability distribution of predictions

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| scikit-learn | ML models & evaluation |
| NLTK | Text preprocessing |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Joblib | Model serialization |
| Pytest | Unit testing |

---

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
