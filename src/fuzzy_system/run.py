"""
Fuzzy system orchestrator.
Usage: python -m src.fuzzy_system.run

Steps:
  1. Load the preprocessed dataset.
  2. Cross-validate the fuzzy inference system across the 5 stratified folds
     (no training step -- see cross_validate_fuzzy for details).
  3. Save the per-fold (y_true, y_pred) results to results/metrics/fuzzy.json.
  4. Print a per-fold and average summary (macro F1, accuracy).
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.cross_validation import cross_validate_fuzzy
from src.evaluation.persistence import save_results
from src.utils.config import DATA_PROCESSED_DIR


def load_preprocessed() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "sentimentdataset_preprocessed.csv"
    return pd.read_csv(path)


def print_summary(results) -> None:
    """Prints per-fold macro F1 / accuracy, plus the average across folds."""
    macro_f1s, accuracies = [], []
    for i, fold in enumerate(results, start=1):
        y_true, y_pred = fold["y_true"], fold["y_pred"]
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1s.append(macro_f1)
        accuracies.append(accuracy)
        print(f"  Fold {i}/{len(results)}: macro F1 = {macro_f1:.3f}, accuracy = {accuracy:.3f}")

    avg_f1 = sum(macro_f1s) / len(macro_f1s)
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\nAverage across {len(results)} folds: macro F1 = {avg_f1:.3f}, accuracy = {avg_acc:.3f}")


def run() -> None:
    df = load_preprocessed()
    print(f"Loaded {len(df)} preprocessed posts.")

    print("\nRunning fuzzy system cross-validation...")
    results = cross_validate_fuzzy(df)

    save_results(results, "fuzzy")

    print("\nFuzzy system results:")
    print_summary(results)


if __name__ == "__main__":
    run()