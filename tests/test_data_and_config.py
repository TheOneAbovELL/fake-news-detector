"""Dataset loading/downloading and configuration tests (no network)."""

import pandas as pd
import pytest

from src import data_loader
from src.config import ConfigError, Settings
from src.data_loader import FAKENEWSNET_FILES, download_fakenewsnet, generate_dataset, load_fakenewsnet


def write_fnn(directory, rows):
    """rows: {file_stem: [titles]} — writes FakeNewsNet-shaped CSVs."""
    for stem in FAKENEWSNET_FILES:
        titles = rows.get(stem, [])
        pd.DataFrame({
            "id": [f"{stem}{i}" for i in range(len(titles))],
            "news_url": [f"example.com/{i}" for i in range(len(titles))],
            "title": titles,
            "tweet_ids": ["1\t2"] * len(titles),
        }).to_csv(directory / f"{stem}.csv", index=False)


class TestFakeNewsNetLoader:
    def test_labels_subsets_and_dedup(self, tmp_path):
        write_fnn(tmp_path, {
            "politifact_fake": ["Pope endorses candidate", "Shared headline", "  Dup Title "],
            "politifact_real": ["Senate passes bill", "Shared headline"],
            "gossipcop_fake": ["dup   title", "Star secretly splits"],
            "gossipcop_real": ["Star attends premiere", None],
        })
        df = load_fakenewsnet(tmp_path)
        assert list(df.columns) == ["text", "label", "subset"]
        assert "Shared headline" not in df["text"].tolist()          # conflicting labels dropped
        assert df["text"].str.lower().str.split().str.join(" ").duplicated().sum() == 0
        assert len(df) == 5                                           # NaN and dups removed
        row = df[df.text == "Pope endorses candidate"].iloc[0]
        assert row.label == 1 and row.subset == "politifact"
        assert df[df.text == "Star attends premiere"].iloc[0].label == 0

    def test_missing_file_has_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="main.py"):
            load_fakenewsnet(tmp_path)


class TestDownload:
    def test_uses_cache_and_downloads_missing_atomically(self, tmp_path, monkeypatch, fake_response):
        write_fnn(tmp_path, {"politifact_fake": ["cached"]})
        for stem in ("politifact_real", "gossipcop_fake", "gossipcop_real"):
            (tmp_path / f"{stem}.csv").unlink()
        requested = []

        def fake_get(url, stream, timeout):
            requested.append(url)
            return fake_response(200, b"id,news_url,title,tweet_ids\n1,u,T,\n")

        import requests
        monkeypatch.setattr(requests, "get", fake_get)
        checksums = download_fakenewsnet(str(tmp_path))
        assert len(requested) == 3 and not any("politifact_fake" in u for u in requested)
        assert set(checksums) == set(FAKENEWSNET_FILES) and all(len(v) == 64 for v in checksums.values())
        assert not list(tmp_path.glob("*.part"))

    def test_failed_download_leaves_no_partial_csv(self, tmp_path, monkeypatch, fake_response):
        import requests
        monkeypatch.setattr(requests, "get", lambda url, stream, timeout: fake_response(500))
        with pytest.raises(requests.exceptions.HTTPError):
            download_fakenewsnet(str(tmp_path))
        assert not list(tmp_path.glob("*.csv"))


class TestSynthetic:
    def test_deterministic_and_no_global_side_effects(self):
        import random
        random.seed(123)
        before = random.random()
        random.seed(123)
        a = generate_dataset(50)
        after = random.random()
        assert before == after                      # global RNG untouched
        pd.testing.assert_frame_equal(a, generate_dataset(50))
        assert data_loader.SEED == 42


class TestSettings:
    def test_defaults(self, monkeypatch):
        for var in ("GROQ_API_KEY", "GROQ_MODEL", "RATE_LIMIT_PER_MINUTE", "MODELS_DIR"):
            monkeypatch.delenv(var, raising=False)
        s = Settings.from_env()
        assert not s.llm_enabled and s.rate_limit_per_minute == 30
        assert s.groq_model == "llama-3.3-70b-versatile"

    def test_env_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "  secret  ")
        monkeypatch.setenv("GROQ_MODEL", "other-model")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
        monkeypatch.setenv("ALLOW_PRIVATE_URLS", "true")
        monkeypatch.setenv("MODELS_DIR", str(tmp_path))
        s = Settings.from_env()
        assert s.groq_api_key == "secret" and s.llm_enabled and s.groq_model == "other-model"
        assert s.rate_limit_per_minute == 0 and s.allow_private_urls and s.models_dir == tmp_path

    def test_key_not_in_repr(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
        assert "super-secret-value" not in repr(Settings.from_env())

    @pytest.mark.parametrize("var,value", [("RATE_LIMIT_PER_MINUTE", "lots"), ("GROQ_TIMEOUT", "0"),
                                           ("ALLOW_PRIVATE_URLS", "maybe"), ("MAX_TEXT_CHARS", "10")])
    def test_invalid_values_rejected(self, monkeypatch, var, value):
        monkeypatch.setenv(var, value)
        with pytest.raises(ConfigError, match=var):
            Settings.from_env()
