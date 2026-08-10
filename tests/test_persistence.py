"""Unit tests for saving/loading cross-validation results as JSON."""
from src.evaluation import persistence


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    """save_results/load_results must reproduce the exact same data
    that was saved -- this is the mechanism src/evaluation/run.py
    relies on to build the final comparison report without retraining
    anything."""
    monkeypatch.setattr(persistence, "RESULTS_METRICS_DIR", tmp_path)

    original = [
        {"y_true": ["Positive", "Negative"], "y_pred": ["Positive", "Neutral"]},
        {"y_true": ["Neutral", "Neutral"], "y_pred": ["Neutral", "Positive"]},
    ]
    persistence.save_results(original, "test_system")
    loaded = persistence.load_results("test_system")

    assert loaded == original


def test_save_creates_results_metrics_directory_if_missing(tmp_path, monkeypatch):
    """train_final.py / run_all.py may be run before results/metrics/
    exists (e.g. a fresh clone) -- save_results must not fail because
    of that."""
    missing_dir = tmp_path / "not_created_yet"
    monkeypatch.setattr(persistence, "RESULTS_METRICS_DIR", missing_dir)

    assert not missing_dir.exists()
    persistence.save_results([{"y_true": [], "y_pred": []}], "test_system")
    assert missing_dir.exists()


def test_load_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    """src/evaluation/run.py relies on this raising FileNotFoundError
    (not, e.g., returning None) so it can give a clear message pointing
    to which upstream script needs to be run first."""
    monkeypatch.setattr(persistence, "RESULTS_METRICS_DIR", tmp_path)

    try:
        persistence.load_results("does_not_exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass