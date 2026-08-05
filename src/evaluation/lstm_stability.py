"""
LSTM run-to-run stability analysis - 6G7V0007 MSc Project
Karyme Nahle Acosta

A single cross_validate_lstm() run uses one fixed seed, so it is fully
reproducible -- but it is still only one sample from a distribution: a
different seed can land on a different (possibly unstable) weight
initialisation, as observed when fold 4 of one run scored macro F1 =
0.365 against ~0.75-0.82 on its other folds.

This module repeats the *entire* 5-fold cross-validation `n_repeats`
times, each with a different seed, and reports the mean and standard
deviation of the average macro F1 across repeats -- a fairer summary of
the LSTM's actual performance than any single run, and a concrete
figure for comparing its stability against the fuzzy system, Naive
Bayes and SVM (none of which vary between runs).

This is optional and NOT part of the main results pipeline
(src/baselines/run_all.py uses a single fixed-seed run, kept as the
primary reported result for reproducibility). Run this separately, and
only if reporting the LSTM's variance is worth the extra training time
(n_repeats full 5-fold runs).
"""
import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.cross_validation import cross_validate_lstm
from src.utils.config import RANDOM_SEED


def repeat_lstm_cv(df, glove_embeddings, n_repeats: int = 5, base_seed: int = RANDOM_SEED):
    """Runs cross_validate_lstm `n_repeats` times, each with seed =
    base_seed + i (i = 0, 1, ..., n_repeats - 1), so every repeat uses a
    genuinely different weight initialisation rather than reproducing
    the same run. Returns a list of dicts, one per repeat, with the
    per-fold results and that repeat's average macro F1."""
    repeats = []
    for i in range(n_repeats):
        seed = base_seed + i
        print(f"\n=== LSTM stability run {i + 1}/{n_repeats} (seed={seed}) ===")
        results = cross_validate_lstm(df, glove_embeddings, seed=seed)

        fold_f1s = [
            f1_score(r["y_true"], r["y_pred"], average="macro", zero_division=0)
            for r in results
        ]
        avg_f1 = float(np.mean(fold_f1s))
        print(f"Run {i + 1} average macro F1: {avg_f1:.3f} (per-fold: {[round(f, 3) for f in fold_f1s]})")

        repeats.append({"seed": seed, "fold_results": results, "fold_f1s": fold_f1s, "avg_f1": avg_f1})
    return repeats


def print_stability_summary(repeats) -> None:
    """Prints the mean and standard deviation of the average macro F1
    across repeats, plus the min/max, to show the spread directly."""
    avg_f1s = [r["avg_f1"] for r in repeats]
    print(f"\nAcross {len(repeats)} full cross-validation runs:")
    print(f"  Mean macro F1 = {np.mean(avg_f1s):.3f}")
    print(f"  Std  macro F1 = {np.std(avg_f1s):.3f}")
    print(f"  Min  macro F1 = {np.min(avg_f1s):.3f}")
    print(f"  Max  macro F1 = {np.max(avg_f1s):.3f}")