"""
Persisting cross-validation results - 6G7V0007 MSc Project
Karyme Nahle Acosta

Saves and loads the per-fold (y_true, y_pred) results produced by the
cross_validate_* functions, as plain JSON, so expensive runs (especially
the LSTM, which retrains from scratch per fold) do not need to be
repeated every time a new metric or analysis is computed from the same
predictions.
"""
import json

from src.utils.config import RESULTS_METRICS_DIR


def save_results(results, name: str) -> None:
    """Saves a list of {"y_true": [...], "y_pred": [...]} dicts (one per
    fold) to results/metrics/<name>.json."""
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_METRICS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def load_results(name: str):
    """Loads results previously saved with save_results()."""
    path = RESULTS_METRICS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)