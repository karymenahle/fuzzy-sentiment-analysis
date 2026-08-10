"""Unit tests for the shared stratified cross-validation folds."""
import pandas as pd

from src.evaluation.folds import get_fold_indices
from src.utils.config import N_FOLDS


def _sample_labels(n_per_class: int = 20) -> pd.Series:
    labels = ["Positive"] * n_per_class + ["Negative"] * n_per_class + ["Neutral"] * n_per_class
    return pd.Series(labels)


def test_returns_n_folds():
    folds = get_fold_indices(_sample_labels())
    assert len(folds) == N_FOLDS


def test_every_sample_appears_in_exactly_one_test_fold():
    """Guards against overlapping or missing test indices, which would
    silently invalidate the whole cross-validation comparison."""
    y = _sample_labels()
    folds = get_fold_indices(y)

    all_test_indices = []
    for _, test_idx in folds:
        all_test_indices.extend(test_idx.tolist())

    assert sorted(all_test_indices) == list(range(len(y)))


def test_train_and_test_indices_do_not_overlap():
    y = _sample_labels()
    folds = get_fold_indices(y)
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_folds_are_reproducible_given_same_seed():
    """cross_validate_fuzzy/naive_bayes/svm/lstm all call this function
    independently and rely on it returning IDENTICAL folds each time --
    otherwise the 4 systems would not be compared on the same data."""
    y = _sample_labels()
    folds_a = get_fold_indices(y)
    folds_b = get_fold_indices(y)

    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert list(train_a) == list(train_b)
        assert list(test_a) == list(test_b)


def test_folds_preserve_class_proportions():
    """Stratification should keep each fold's class balance close to
    the overall balance, even with the imbalanced real class
    distribution (~460 Positive / ~185 Negative / ~61 Neutral)."""
    y = pd.Series(["Positive"] * 46 + ["Negative"] * 18 + ["Neutral"] * 6)
    folds = get_fold_indices(y)

    for _, test_idx in folds:
        fold_labels = y.iloc[test_idx]
        # every fold must include at least one example of each class
        assert set(fold_labels.unique()) == {"Positive", "Negative", "Neutral"}