"""
Borderline-case analysis - 6G7V0007 MSc Project
Karyme Nahle Acosta

Joins the true label, the fuzzy system's continuous sentiment_score, and
every system's prediction into a single table indexed by the original
post, then compares how each system performs specifically on borderline
(ambiguous) posts versus clear-cut ones.

Saved cross-validation results are organised by fold, not by the
original row order, so the same stratified folds (same random seed) are
regenerated here purely to recover which original row corresponds to
each entry in the saved results.
"""
import pandas as pd

from src.evaluation.folds import get_fold_indices
from src.evaluation.persistence import load_results
from src.fuzzy_system.engine import classify_score, infer_sentiment
from src.fuzzy_system.inputs import compute_fuzzy_inputs


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with one row per post (706 rows), columns:
    text, true_label, fuzzy_score, fuzzy_pred, naive_bayes_pred,
    svm_pred, lstm_pred."""
    folds = get_fold_indices(df["sentiment_3class"])

    saved = {
        "naive_bayes": load_results("naive_bayes"),
        "svm": load_results("svm"),
        "fuzzy": load_results("fuzzy"),
        "lstm": load_results("lstm"),
    }

    rows = []
    for fold_num, (_, test_idx) in enumerate(folds):
        for position_in_fold, original_pos in enumerate(test_idx):
            row = df.iloc[original_pos]
            rows.append({
                "text": row["Text"],
                "true_label": row["sentiment_3class"],
                "naive_bayes_pred": saved["naive_bayes"][fold_num]["y_pred"][position_in_fold],
                "svm_pred": saved["svm"][fold_num]["y_pred"][position_in_fold],
                "fuzzy_pred": saved["fuzzy"][fold_num]["y_pred"][position_in_fold],
                "lstm_pred": saved["lstm"][fold_num]["y_pred"][position_in_fold],
            })

    table = pd.DataFrame(rows)

    # The fuzzy score itself does not depend on folds (no training), so
    # it is computed directly and merged in by text.
    scores = []
    for text in table["text"]:
        inputs = compute_fuzzy_inputs(text)
        score = infer_sentiment(
            inputs["positive_intensity"], inputs["negative_intensity"], inputs["subjectivity"]
        )
        scores.append(score)
    table["fuzzy_score"] = scores

    return table


def borderline_mask(table: pd.DataFrame, threshold: float = 0.25) -> pd.Series:
    """Flags posts as borderline when the fuzzy score's absolute value
    is below `threshold`. 0.25 matches the width of the Neutral output
    membership function, i.e. posts the fuzzy system itself treats as
    at least partially Neutral.

    Note: this definition is not neutral between systems -- it uses the
    fuzzy system's own score to decide which posts count as "ambiguous",
    which can bias the comparison in either direction. See
    disagreement_mask for an alternative that does not have this
    problem."""
    return table["fuzzy_score"].abs() <= threshold


def disagreement_mask(table: pd.DataFrame) -> pd.Series:
    """Flags posts where the 4 systems do not unanimously agree on a
    prediction. Unlike borderline_mask, this does not rely on any single
    system's own confidence/score, so it is a fairer proxy for "this post
    is genuinely hard to classify" when comparing the 4 systems against
    each other."""
    preds = table[["fuzzy_pred", "naive_bayes_pred", "svm_pred", "lstm_pred"]]
    return preds.nunique(axis=1) > 1


def _accuracy_by_mask(table: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    systems = {
        "Fuzzy": "fuzzy_pred",
        "Naive Bayes": "naive_bayes_pred",
        "SVM": "svm_pred",
        "LSTM": "lstm_pred",
    }
    rows = []
    for name, pred_col in systems.items():
        correct = table["true_label"] == table[pred_col]
        rows.append({
            "system": name,
            "n_ambiguous": int(mask.sum()),
            "accuracy_ambiguous": correct[mask].mean(),
            "n_clear": int((~mask).sum()),
            "accuracy_clear": correct[~mask].mean(),
        })
    return pd.DataFrame(rows)


def compare_accuracy_by_zone(table: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Accuracy comparison using borderline_mask (fuzzy-score-based
    definition of ambiguity)."""
    return _accuracy_by_mask(table, borderline_mask(table, threshold))


def compare_accuracy_by_disagreement(table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy comparison using disagreement_mask (model-agreement-based
    definition of ambiguity) -- the fairer version, since it does not
    use any one system's own score to decide what counts as ambiguous."""
    return _accuracy_by_mask(table, disagreement_mask(table))