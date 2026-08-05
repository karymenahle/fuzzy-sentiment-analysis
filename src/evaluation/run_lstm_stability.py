"""
LSTM stability orchestrator (optional).
Usage: python -m src.evaluation.run_lstm_stability

Repeats the full 5-fold LSTM cross-validation several times, each with a
different seed, and reports the mean/std of the average macro F1 across
runs. This is separate from the main pipeline (src/baselines/run_all.py)
and is only worth running if you want a variance figure to report
alongside the single reproducible LSTM result -- it takes roughly
n_repeats times as long as a single run of cross_validate_lstm.

Saves each repeat's raw fold results to
results/metrics/lstm_stability_repeats.json, and its summary (mean, std,
min, max, per-repeat average F1) to
results/metrics/lstm_stability_summary.json.
"""
import json

from src.baselines.glove_embeddings import load_glove_embeddings
from src.baselines.run_all import load_preprocessed
from src.evaluation.lstm_stability import print_stability_summary, repeat_lstm_cv
from src.utils.config import DATA_EMBEDDINGS_DIR, GLOVE_FILENAME, RESULTS_METRICS_DIR

N_REPEATS = 5


def run() -> None:
    df = load_preprocessed()
    print(f"Loaded {len(df)} preprocessed posts.")

    print("\nLoading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings(DATA_EMBEDDINGS_DIR / GLOVE_FILENAME)
    print(f"Loaded {len(glove_embeddings)} GloVe vectors.")

    repeats = repeat_lstm_cv(df, glove_embeddings, n_repeats=N_REPEATS)
    print_stability_summary(repeats)

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_METRICS_DIR / "lstm_stability_repeats.json", "w") as f:
        json.dump(repeats, f, indent=2)

    summary = {
        "n_repeats": N_REPEATS,
        "avg_f1_per_repeat": [r["avg_f1"] for r in repeats],
        "seeds": [r["seed"] for r in repeats],
    }
    with open(RESULTS_METRICS_DIR / "lstm_stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {RESULTS_METRICS_DIR / 'lstm_stability_repeats.json'}")
    print(f"Saved: {RESULTS_METRICS_DIR / 'lstm_stability_summary.json'}")


if __name__ == "__main__":
    run()