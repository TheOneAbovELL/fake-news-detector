from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent
requirements = [
    line.strip() for line in (here / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="truthlens",
    version="2.0.0",
    author="Omjee R Giri",
    description="TruthLens — news credibility checker (TF-IDF + logistic regression, optional LLM)",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={"dev": ["pytest==8.3.4", "ruff==0.8.4"], "stemming": ["nltk>=3.8"]},
)
