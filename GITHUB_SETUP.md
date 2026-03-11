# 🚀 Publishing to GitHub — Step-by-Step Guide

Follow these steps **after** running `python main.py` successfully on your machine.

---

## Step 1: Initialize Git repository

```bash
cd fake-news-detector
git init
git add .
git commit -m "Initial commit: Fake News Detection pipeline — BIT Mesra 2024"
```

---

## Step 2: Create GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `fake-news-detector`
3. Description: `Fake News Detection using TF-IDF + Logistic Regression & SVM — BIT Mesra 2024`
4. Set to **Public**
5. ❌ Do NOT initialize with README (we already have one)
6. Click **Create repository**

---

## Step 3: Push to GitHub

```bash
git remote add origin https://github.com/<Omjee R Giri>/fake-news-detector.git
git branch -M main
git push -u origin main
```

Replace `<YOUR_USERNAME>` with your GitHub username.

---

## Step 4: Add topics/tags (optional but recommended)

In your GitHub repo → ⚙️ Settings → Topics, add:
```
machine-learning  nlp  fake-news-detection  scikit-learn  tfidf  python  classification
```

---

## ✅ What gets published

```
fake-news-detector/
├── src/                ← Full source code
├── tests/              ← 29 unit tests
├── reports/figures/    ← All generated plots
├── main.py             ← Run-everything entry point
├── requirements.txt
├── setup.py
├── README.md           ← Professional documentation
└── LICENSE
```

> Note: `models/*.pkl` and `data/*.csv` are excluded via `.gitignore` (generated at runtime).

---

## 💡 Tips

- Add a **screenshot** of your figures to the README for visual appeal
- Star your own repo and share the link on LinkedIn
- Consider adding a `CONTRIBUTING.md` if you want others to contribute
