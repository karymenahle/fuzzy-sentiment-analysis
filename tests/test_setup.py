"""Smoke tests to confirm the project skeleton is wired up correctly."""
from src.utils import config


def test_directories_exist():
    assert config.DATA_RAW_DIR.parent.exists()
    assert config.RESULTS_FIGURES_DIR.parent.exists()


def test_sentiment_labels():
    assert config.SENTIMENT_LABELS == ["positive", "negative", "neutral"]


def test_random_seed_is_int():
    assert isinstance(config.RANDOM_SEED, int)
