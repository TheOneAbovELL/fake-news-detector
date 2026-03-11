from setuptools import setup, find_packages

setup(
    name="fake-news-detector",
    version="1.0.0",
    author="BIT Mesra",
    description="Fake News Detection using TF-IDF + Logistic Regression & SVM",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "jupyter>=1.0.0"],
    },
)
