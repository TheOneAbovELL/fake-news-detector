"""
data_loader.py
--------------
Dataset generation and loading for the Fake News Detection pipeline.

Since the original WELFake/LIAR datasets require registration, this module
generates a realistic synthetic dataset with authentic linguistic patterns
drawn from research literature on fake vs. real news characteristics.
"""

import os
import random
import pandas as pd
import numpy as np
from pathlib import Path


# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Linguistic building blocks ──────────────────────────────────────────────

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


def _fill_template(template: str) -> str:
    """Fill a headline template with random components."""
    return template.format(
        source=random.choice(SOURCES),
        subject=random.choice(SUBJECTS),
        context=random.choice(EVENTS),
        policy=random.choice(TOPICS),
        event=random.choice(EVENTS),
        finding=random.choice(FINDINGS),
        cause=random.choice(TOPICS),
        official=random.choice(OFFICIALS),
        action=random.choice(["resign", "testify", "respond", "act", "intervene"]),
        country=random.choice(["US", "UK", "Germany", "France", "Japan", "India"]),
        document=random.choice(["white paper", "report", "statement", "brief"]),
        topic=random.choice(TOPICS),
        study=random.choice(["study", "trial", "meta-analysis", "survey"]),
        organization=random.choice(["WHO", "IMF", "UN", "OECD", "WTO"]),
        decision=random.choice(["in favor", "against", "to uphold", "to overturn"]),
        partner=random.choice(["EU", "China", "NATO", "ASEAN", "G20"]),
        outcome=random.choice(["reduce costs", "save lives", "boost growth"]),
        incident=random.choice(["data breach", "protest", "accident", "outage"]),
        company=random.choice(["major tech firm", "leading bank", "energy company"]),
        venue=random.choice(["Davos", "G20 summit", "UN Assembly", "Brussels"]),
        conspiracy=random.choice(CONSPIRACIES),
        celebrity=random.choice(CELEBRITIES),
        disease=random.choice(["cancer", "diabetes", "Alzheimer's", "arthritis"]),
        scandal=random.choice(["financial misconduct", "cover-up", "secret deal"]),
        government=random.choice(["Deep State", "shadow government", "elites", "globalists"]),
    )


def generate_dataset(n_samples: int = 5000, save_path: str = None) -> pd.DataFrame:
    """
    Generate a realistic fake/real news headline dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples (split 50/50 fake/real).
    save_path : str, optional
        If provided, saves CSV to this path.

    Returns
    -------
    pd.DataFrame with columns: ['text', 'label'] where label 0=real, 1=fake
    """
    print(f"[DataLoader] Generating {n_samples} news headlines...")

    half = n_samples // 2
    records = []

    # Generate REAL headlines (label = 0)
    for _ in range(half):
        template = random.choice(REAL_TEMPLATES)
        headline = _fill_template(template)
        # Real news: slightly longer, more measured
        if random.random() > 0.7:
            headline = headline + ", " + random.choice([
                "officials say", "sources confirm", "data shows",
                "report finds", "experts note",
            ])
        records.append({"text": headline, "label": 0})

    # Generate FAKE headlines (label = 1)
    for _ in range(half):
        template = random.choice(FAKE_TEMPLATES)
        headline = _fill_template(template)
        records.append({"text": headline, "label": 1})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DataLoader] Dataset saved → {save_path}")

    print(f"[DataLoader] Dataset ready: {len(df)} rows | "
          f"Real: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")
    return df


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"
    return df
