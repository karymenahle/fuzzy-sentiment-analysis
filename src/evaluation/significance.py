"""
Statistical significance testing - 6G7V0007 MSc Project
Karyme Nahle Acosta

Two complementary tests for comparing a pair of systems evaluated on the
same 5 stratified folds:
  - paired_ttest_macro_f1: compares the 5 per-fold macro F1 scores of
    two systems, testing whether the mean difference is significant.
  - mcnemar_test: compares predictions instance-by-instance across the
    pooled test folds, testing whether the two systems' errors are
    systematically different rather than incidental.
"""
import numpy as np
from scipy.stats import binomtest, chi2, ttest_rel
from sklearn.metrics import f1_score


def paired_ttest_macro_f1(results_a, results_b):
    """Paired t-test over per-fold macro F1. Valid here because both
    systems were evaluated on the exact same 5 folds (same random
    seed), so each pair of F1 values corresponds to the same held-out
    data."""
    f1_a = [f1_score(r["y_true"], r["y_pred"], average="macro", zero_division=0) for r in results_a]
    f1_b = [f1_score(r["y_true"], r["y_pred"], average="macro", zero_division=0) for r in results_b]
    statistic, pvalue = ttest_rel(f1_a, f1_b)
    return {
        "f1_a": f1_a, "f1_b": f1_b,
        "mean_diff": float(np.mean(f1_a) - np.mean(f1_b)),
        "statistic": float(statistic), "pvalue": float(pvalue),
    }


def _pooled_correctness(results):
    y_true, y_pred = [], []
    for r in results:
        y_true.extend(r["y_true"])
        y_pred.extend(r["y_pred"])
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    return correct, y_true


def mcnemar_test(results_a, results_b):
    """McNemar's test on predictions pooled across all 5 folds. Valid
    here because both systems were cross-validated with the same
    stratified folds (same random seed), so once each system's 5 test
    folds are concatenated in order, they cover the full dataset
    exactly once, in the same instance order for both systems."""
    correct_a, y_true_a = _pooled_correctness(results_a)
    correct_b, y_true_b = _pooled_correctness(results_b)
    if y_true_a != y_true_b:
        raise ValueError(
            "y_true sequences differ between systems -- were the same folds used for both?"
        )

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_correct = int(np.sum(correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    table = [[both_correct, a_only], [b_only, neither]]

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"table": table, "statistic": None, "pvalue": 1.0, "method": "no discordant pairs"}

    if n_discordant < 25:
        pvalue = binomtest(min(a_only, b_only), n_discordant, 0.5).pvalue
        return {"table": table, "statistic": None, "pvalue": pvalue, "method": "exact binomial"}

    statistic = (abs(a_only - b_only) - 1) ** 2 / n_discordant
    pvalue = 1 - chi2.cdf(statistic, df=1)
    return {"table": table, "statistic": statistic, "pvalue": pvalue, "method": "chi-square (continuity corrected)"}