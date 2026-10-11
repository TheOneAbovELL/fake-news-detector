"""
data_loader.py
--------------
Dataset acquisition and loading for the Fake News Detection pipeline.

Two sources are supported:

* ``fakenewsnet`` (default, used to train the served model) — ~23k real news
  headlines labelled through PolitiFact and GossipCop fact-checks
  (Shu et al., "FakeNewsNet", 2018; https://github.com/KaiDMML/FakeNewsNet).
* ``synthetic`` — template-generated headlines. Only useful as a fast, offline
  fixture for tests and smoke runs: a model trained on it learns the templates,
  not misinformation, and must never be served as a detector.
"""

import hashlib
import os
import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SEED = 42

FAKENEWSNET_BASE_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
FAKENEWSNET_FILES = {
    # file stem: (subset, label)  — label 1 = fake, 0 = real
    "politifact_fake": ("politifact", 1),
    "politifact_real": ("politifact", 0),
    "gossipcop_fake": ("gossipcop", 1),
    "gossipcop_real": ("gossipcop", 0),
}

# ── Synthetic templates (test fixture only) ─────────────────────────────────

REAL_TEMPLATES = [
    "{source} reports {subject} amid {context}",
    "{subject} announces new {policy} following {event}",
    "Study finds {finding} linked to {cause}",
    "{official} confirms {action} after {event}",
    "{country} government releases {document} on {topic}",
    "Markets react to {event} as {subject} {action}",
    "Scientists discover {finding} in landmark {study}",
    "{organization} publishes annual report on {topic}",
    "Court rules {decision} in case involving {subject}",
    "{country} reaches agreement with {partner} on {topic}",
    "New research suggests {finding} could {outcome}",
    "Officials investigate {incident} following {event}",
    "{company} releases quarterly earnings above expectations",
    "Global leaders meet to discuss {topic} at {venue}",
    "Health authorities issue guidelines on {topic}",
]

FAKE_TEMPLATES = [
    "BREAKING: {subject} secretly {action} — mainstream media silent!",
    "EXPOSED: The truth about {topic} they don't want you to know",
    "{celebrity} CONFIRMS {conspiracy} in leaked video",
    "Doctors HATE this: {subject} cures {disease} overnight",
    "SHOCKING: {official} caught {action} on camera",
    "URGENT: {government} planning to {conspiracy} — share before deleted",
    "{subject} ADMITS to {scandal} in bombshell confession",
    "Scientists BAFFLED by {subject} — establishment hiding the truth",
    "You won't BELIEVE what {official} just said about {topic}",
    "PROOF: {conspiracy} confirmed — elites in panic",
    "Deep state operative REVEALS plot to {conspiracy}",
    "{celebrity} exposes {scandal} — Hollywood in shock",
    "MUST WATCH: {subject} destroys {topic} with one statement",
    "They're HIDING the cure for {disease} — here's the truth",
    "LEAKED: Secret documents prove {conspiracy}",
]

SUBJECTS = [
    "the Federal Reserve", "WHO", "NASA", "the Pentagon", "EU Commission",
    "the White House", "UN Security Council", "CDC", "FBI", "World Bank",
    "the Senate", "Supreme Court", "G7 leaders", "OPEC", "IMF",
]

TOPICS = [
    "climate policy", "vaccine safety", "economic growth", "cybersecurity",
    "trade agreements", "public health", "immigration reform", "AI regulation",
    "renewable energy", "inflation", "healthcare reform", "nuclear policy",
    "education funding", "water scarcity", "digital privacy",
]

OFFICIALS = [
    "senior officials", "government spokesperson", "agency director",
    "committee chairman", "cabinet minister", "department secretary",
    "federal investigators", "intelligence officials", "health experts",
]

EVENTS = [
    "recent summit", "emergency meeting", "policy review", "leaked memo",
    "congressional hearing", "international conference", "annual assessment",
    "quarterly review", "public inquiry", "independent audit",
]

FINDINGS = [
    "significant correlation", "increased risk factors", "measurable improvement",
    "substantial evidence", "notable decline", "consistent patterns",
    "unexpected outcomes", "strong indicators", "preliminary results",
]

CONSPIRACIES = [
    "control the food supply", "microchip the population",
    "shut down free speech", "collapse the economy on purpose",
    "install surveillance in homes", "poison the water",
    "rig the electoral system", "eliminate cash currency",
]

CELEBRITIES = [
    "A-list actor", "famous singer", "retired athlete",
    "tech billionaire", "reality TV star", "social media influencer",
]

SOURCES = [
    "Reuters", "Associated Press", "BBC", "The Guardian",
    "NPR", "The Wall Street Journal", "Financial Times",
]


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a headline template with random components."""
    c = rng.choice
    return template.format(
        source=c(SOURCES),
        subject=c(SUBJECTS),
        context=c(EVENTS),
        policy=c(TOPICS),
        event=c(EVENTS),
        finding=c(FINDINGS),
        cause=c(TOPICS),
        official=c(OFFICIALS),
        action=c(["resign", "testify", "respond", "act", "intervene"]),
        country=c(["US", "UK", "Germany", "France", "Japan", "India"]),
        document=c(["white paper", "report", "statement", "brief"]),
        topic=c(TOPICS),
        study=c(["study", "trial", "meta-analysis", "survey"]),
        organization=c(["WHO", "IMF", "UN", "OECD", "WTO"]),
        decision=c(["in favor", "against", "to uphold", "to overturn"]),
        partner=c(["EU", "China", "NATO", "ASEAN", "G20"]),
        outcome=c(["reduce costs", "save lives", "boost growth"]),
        incident=c(["data breach", "protest", "accident", "outage"]),
        company=c(["major tech firm", "leading bank", "energy company"]),
        venue=c(["Davos", "G20 summit", "UN Assembly", "Brussels"]),
        conspiracy=c(CONSPIRACIES),
        celebrity=c(CELEBRITIES),
        disease=c(["cancer", "diabetes", "Alzheimer's", "arthritis"]),
        scandal=c(["financial misconduct", "cover-up", "secret deal"]),
        government=c(["Deep State", "shadow government", "elites", "globalists"]),
    )


def generate_dataset(n_samples: int = 5000, save_path: str = None,
                     seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic, template-based headline dataset (test fixture only).

    Returns
    -------
    pd.DataFrame with columns: ['text', 'label'] where label 0=real, 1=fake
    """
    rng = random.Random(seed)
    half = n_samples // 2
    records = []

    for _ in range(half):
        headline = _fill_template(rng.choice(REAL_TEMPLATES), rng)
        if rng.random() > 0.7:
            headline = headline + ", " + rng.choice([
                "officials say", "sources confirm", "data shows",
                "report finds", "experts note",
            ])
        records.append({"text": headline, "label": 0})

    for _ in range(n_samples - half):
        records.append({"text": _fill_template(rng.choice(FAKE_TEMPLATES), rng), "label": 1})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


# ── FakeNewsNet ─────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_fakenewsnet(data_dir: str, force: bool = False,
                         timeout: float = 60.0) -> Dict[str, str]:
    """
    Download the four FakeNewsNet CSVs into ``data_dir``.

    Files already present are reused (pass ``force=True`` to re-download).
    Each file is streamed to a ``.part`` file and renamed only when complete,
    so an interrupted download never leaves a truncated CSV behind.

    Returns {file_stem: sha256} for provenance tracking.
    """
    import requests

    target = Path(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for stem in FAKENEWSNET_FILES:
        path = target / f"{stem}.csv"
        if path.exists() and path.stat().st_size > 0 and not force:
            print(f"[DataLoader] Using cached {path}")
        else:
            url = f"{FAKENEWSNET_BASE_URL}/{stem}.csv"
            print(f"[DataLoader] Downloading {url}")
            part = path.with_suffix(".csv.part")
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            os.replace(part, path)
        checksums[stem] = _sha256(path)
    return checksums


def load_fakenewsnet(data_dir: str) -> pd.DataFrame:
    """
    Load FakeNewsNet headlines as a clean, de-duplicated DataFrame.

    * keeps only the title (the URL/domain would leak the label)
    * drops titles that appear with *both* labels (unresolvable noise)
    * keeps one copy of each remaining duplicate title, so the same headline
      can never land in both the train and the test split

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'subset'] (label 1 = fake)
    """
    frames = []
    for stem, (subset, label) in FAKENEWSNET_FILES.items():
        path = Path(data_dir) / f"{stem}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python main.py` (downloads automatically)"
            )
        part = pd.read_csv(path, usecols=["title"], dtype={"title": str})
        part = part.rename(columns={"title": "text"})
        part["label"] = label
        part["subset"] = subset
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    key = df["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    labels_per_title = df.groupby(key)["label"].nunique()
    conflicting = set(labels_per_title[labels_per_title > 1].index)
    keep = ~key.isin(conflicting) & ~key.duplicated()
    df = df[keep].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"[DataLoader] FakeNewsNet: {len(df):,} unique headlines | "
          f"real: {(df.label == 0).sum():,} | fake: {(df.label == 1).sum():,} | "
          f"dropped {len(conflicting)} conflicting titles")
    return df


def load_dataset(path: str, required: Optional[tuple] = ("text", "label")) -> pd.DataFrame:
    """Load a labelled dataset from CSV (must contain 'text' and 'label' columns)."""
    df = pd.read_csv(path)
    missing = [c for c in (required or ()) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must have columns {list(required)}; missing {missing}")
    return df
