"""
Stratified k-fold cross-validation setup - 6G7V0007 MSc Project
Karyme Nahle Acosta

Generates the 5 stratified folds used to evaluate all 4 systems (fuzzy,
Naive Bayes, SVM, LSTM) under identical train/test splits, so that any
difference in performance is attributable to the classification
approach rather than to which posts happened to land in which fold.
"""
from sklearn.model_selection import StratifiedKFold

from src.utils.config import N_FOLDS, RANDOM_SEED


def get_fold_indices(y):
    """Returns a list of (train_idx, test_idx) tuples, one per fold.
    `y` is the full array/Series of class labels used purely to
    determine the stratification -- the actual features passed to each
    model are indexed separately using the same indices."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(X=y, y=y))