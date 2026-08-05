"""
Evaluation orchestrator (final comparison report).
Usage: python -m src.evaluation.run

Steps:
  1. Load the preprocessed dataset and the 4 saved cross-validation results
     (fuzzy, naive_bayes, svm, lstm -- produced by src/fuzzy_system/run.py
     and src/baselines/run_all.py).
  2. Print a per-system metrics summary (macro F1, accuracy).
  3. Run paired t-test and McNemar's test for all 6 possible pairs among
     the 4 systems.
  4. Run the borderline-case analysis (both the fuzzy-score-based and the
     disagreement-based definitions of "ambiguous").
  5. Save everything to results/metrics/final_report.json.
"""
import itertools
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.borderline import (
    build_comparison_table,
    compare_accuracy_by_disagreement,
    compare_accuracy_by_zone,
)
from src.evaluation.persistence import load_results
from src.evaluation.significance import mcnemar_test, paired_ttest_macro_f1
from src.utils.config import DATA_PROCESSED_DIR, RESULTS_METRICS_DIR

SYSTEM_NAMES = ["fuzzy", "naive_bayes", "svm", "lstm"]


def load_preprocessed() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "sentimentdataset_preprocessed.csv"
    return pd.read_csv(path)


def load_all_results() -> dict:
    """Loads the 4 systems' saved cross-validation results. Raises a
    clear error naming the missing file if one of the two upstream
    scripts (fuzzy_system/run.py, baselines/run_all.py) has not been
    run yet."""
    results = {}
    for name in SYSTEM_NAMES:
        try:
            results[name] = load_results(name)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing results/metrics/{name}.json -- run "
                f"'python -m src.fuzzy_system.run' and "
                f"'python -m src.baselines.run_all' first."
            ) from e
    return results


def compute_metrics_summary(all_results: dict) -> dict:
    """Returns {system_name: {"macro_f1": [...5 folds...], "accuracy":
    [...5 folds...], "avg_macro_f1": x, "avg_accuracy": y}}."""
    summary = {}
    for name, results in all_results.items():
        macro_f1s = [f1_score(r["y_true"], r["y_pred"], average="macro", zero_division=0) for r in results]
        accuracies = [accuracy_score(r["y_true"], r["y_pred"]) for r in results]
        summary[name] = {
            "macro_f1_per_fold": macro_f1s,
            "accuracy_per_fold": accuracies,
            "avg_macro_f1": sum(macro_f1s) / len(macro_f1s),
            "avg_accuracy": sum(accuracies) / len(accuracies),
        }
    return summary


def print_metrics_summary(summary: dict) -> None:
    print(f"{'System':<15}{'Avg macro F1':<15}{'Avg accuracy':<15}")
    for name, m in summary.items():
        print(f"{name:<15}{m['avg_macro_f1']:<15.3f}{m['avg_accuracy']:<15.3f}")


def run_significance_tests(all_results: dict) -> dict:
    """Runs paired t-test (macro F1) and McNemar's test for all 6
    possible pairs among the 4 systems."""
    pair_results = {}
    for name_a, name_b in itertools.combinations(SYSTEM_NAMES, 2):
        pair_key = f"{name_a}_vs_{name_b}"
        print(f"\n--- {name_a} vs {name_b} ---")

        ttest = paired_ttest_macro_f1(all_results[name_a], all_results[name_b])
        print(f"  Paired t-test: mean diff = {ttest['mean_diff']:.3f}, "
              f"statistic = {ttest['statistic']:.3f}, p = {ttest['pvalue']:.3f}")

        try:
            mcnemar = mcnemar_test(all_results[name_a], all_results[name_b])
            print(f"  McNemar's test ({mcnemar['method']}): p = {mcnemar['pvalue']:.3f}, "
                  f"table = {mcnemar['table']}")
        except ValueError as e:
            print(f"  McNemar's test skipped: {e}")
            mcnemar = None

        pair_results[pair_key] = {"ttest": ttest, "mcnemar": mcnemar}
    return pair_results


def run_borderline_analysis(df: pd.DataFrame) -> dict:
    """Runs both definitions of "ambiguous" (fuzzy-score-based and
    disagreement-based) and returns their comparison tables as dicts."""
    print("\nBuilding comparison table (this re-runs fuzzy inference on "
          "every post, but does not retrain any classifier)...")
    table = build_comparison_table(df)

    print("\nAccuracy by zone (fuzzy-score-based definition of ambiguous):")
    by_zone = compare_accuracy_by_zone(table)
    print(by_zone.to_string(index=False))

    print("\nAccuracy by zone (disagreement-based definition of ambiguous):")
    by_disagreement = compare_accuracy_by_disagreement(table)
    print(by_disagreement.to_string(index=False))

    return {
        "by_fuzzy_score": by_zone.to_dict(orient="records"),
        "by_disagreement": by_disagreement.to_dict(orient="records"),
    }


def run() -> None:
    df = load_preprocessed()
    print(f"Loaded {len(df)} preprocessed posts.")

    all_results = load_all_results()

    print("\n=== Metrics summary ===")
    metrics_summary = compute_metrics_summary(all_results)
    print_metrics_summary(metrics_summary)

    print("\n=== Statistical significance tests ===")
    significance_results = run_significance_tests(all_results)

    print("\n=== Borderline-case analysis ===")
    borderline_results = run_borderline_analysis(df)

    report = {
        "metrics_summary": metrics_summary,
        "significance_tests": significance_results,
        "borderline_analysis": borderline_results,
    }
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_METRICS_DIR / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    run()