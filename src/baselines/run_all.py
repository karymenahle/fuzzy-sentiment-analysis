"""
Baselines orchestrator.
Usage: python -m src.baselines.run_all

Steps:
  1. Load the preprocessed dataset.
  2. Cross-validate Naive Bayes across the 5 stratified folds and save results.
  3. Cross-validate SVM across the same folds and save results.
  4. Load the GloVe embeddings once.
  5. Cross-validate the LSTM (reusing the loaded GloVe embeddings) and save results.

Naive Bayes and SVM run first since they are fast: if the (much slower) LSTM
step fails or is interrupted, their results are already saved.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.baselines.glove_embeddings import load_glove_embeddings
from src.evaluation.cross_validation import (
    cross_validate_lstm,
    cross_validate_naive_bayes,
    cross_validate_svm,
)
from src.evaluation.persistence import save_results
from src.utils.config import DATA_EMBEDDINGS_DIR, DATA_PROCESSED_DIR, GLOVE_FILENAME, RANDOM_SEED
from src.utils.seed import set_global_seed


def load_preprocessed() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "sentimentdataset_preprocessed.csv"
    return pd.read_csv(path)


def print_summary(results, name: str) -> None:
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
    print(f"\n{name} average across {len(results)} folds: macro F1 = {avg_f1:.3f}, accuracy = {avg_acc:.3f}")


def run() -> None:
    set_global_seed(RANDOM_SEED)
    df = load_preprocessed()
    print(f"Loaded {len(df)} preprocessed posts.")

    print("\nRunning Naive Bayes cross-validation...")
    nb_results = cross_validate_naive_bayes(df)
    save_results(nb_results, "naive_bayes")
    print("\nNaive Bayes results:")
    print_summary(nb_results, "Naive Bayes")

    print("\nRunning SVM cross-validation...")
    svm_results = cross_validate_svm(df)
    save_results(svm_results, "svm")
    print("\nSVM results:")
    print_summary(svm_results, "SVM")

    print("\nLoading GloVe embeddings...")
    glove_path = DATA_EMBEDDINGS_DIR / GLOVE_FILENAME
    glove_embeddings = load_glove_embeddings(glove_path)
    print(f"Loaded {len(glove_embeddings)} GloVe vectors.")

    print("\nRunning LSTM cross-validation...")
    lstm_results = cross_validate_lstm(df, glove_embeddings)
    save_results(lstm_results, "lstm")
    print("\nLSTM results:")
    print_summary(lstm_results, "LSTM")


if __name__ == "__main__":
    run()