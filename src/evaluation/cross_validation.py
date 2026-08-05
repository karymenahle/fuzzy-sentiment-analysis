"""
Cross-validation loop for the classical baselines - 6G7V0007 MSc Project
Karyme Nahle Acosta

Runs Naive Bayes and SVM across the 5 stratified folds. The TF-IDF
vectorizer is fit separately within each fold, using only that fold's
training data, so no information from the test portion leaks into the
vocabulary before evaluation.
"""
from sklearn.model_selection import train_test_split

from src.baselines.naive_bayes import train_naive_bayes
from src.baselines.svm import train_svm
from src.baselines.vectorizer import build_tfidf_vectorizer
from src.baselines.lstm_data import (
    build_tokenizer, encode_labels, texts_to_padded_sequences, INT_TO_LABEL,
)
from src.baselines.glove_embeddings import build_embedding_matrix
from src.baselines.lstm_model import build_lstm_model, train_lstm
from src.evaluation.folds import get_fold_indices
from src.fuzzy_system.engine import classify_score, infer_sentiment
from src.fuzzy_system.inputs import compute_fuzzy_inputs
from src.utils.config import EMBEDDING_DIM, RANDOM_SEED
from src.utils.seed import set_global_seed


def cross_validate_naive_bayes(df):
    """Returns a list of dicts, one per fold, each with the true and
    predicted labels for that fold's test portion."""
    folds = get_fold_indices(df["sentiment_3class"])
    results = []
    for train_idx, test_idx in folds:
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        vectorizer = build_tfidf_vectorizer()
        X_train = vectorizer.fit_transform(train_df["tfidf_text"])
        X_test = vectorizer.transform(test_df["tfidf_text"])

        model = train_naive_bayes(X_train, train_df["sentiment_3class"])
        preds = model.predict(X_test)

        results.append({
            "y_true": test_df["sentiment_3class"].tolist(),
            "y_pred": preds.tolist(),
        })
    return results


def cross_validate_svm(df):
    """Same idea as cross_validate_naive_bayes, but for the linear-kernel
    SVM baseline."""
    folds = get_fold_indices(df["sentiment_3class"])
    results = []
    for train_idx, test_idx in folds:
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        vectorizer = build_tfidf_vectorizer()
        X_train = vectorizer.fit_transform(train_df["tfidf_text"])
        X_test = vectorizer.transform(test_df["tfidf_text"])

        model = train_svm(X_train, train_df["sentiment_3class"])
        preds = model.predict(X_test)

        results.append({
            "y_true": test_df["sentiment_3class"].tolist(),
            "y_pred": preds.tolist(),
        })
    return results


def cross_validate_fuzzy(df):
    """Same fold structure as the other systems, for a fair comparison,
    but the fuzzy system has no training step: its rules and membership
    functions are fixed by design, not learned from data. Each fold's
    test portion is fuzzified and classified directly, using the
    `lexicon_text` branch (VADER/TextBlob need case and punctuation
    intact)."""
    folds = get_fold_indices(df["sentiment_3class"])
    results = []
    for train_idx, test_idx in folds:
        test_df = df.iloc[test_idx]

        preds = []
        for lexicon_text in test_df["lexicon_text"]:
            fuzzy_inputs = compute_fuzzy_inputs(lexicon_text)
            score = infer_sentiment(
                fuzzy_inputs["positive_intensity"],
                fuzzy_inputs["negative_intensity"],
                fuzzy_inputs["subjectivity"],
            )
            preds.append(classify_score(score))

        results.append({
            "y_true": test_df["sentiment_3class"].tolist(),
            "y_pred": preds,
        })
    return results


def cross_validate_lstm(df, glove_embeddings, max_len: int = 30, epochs: int = 30,
                         seed: int = RANDOM_SEED):
    """Same fold structure as the other systems. `glove_embeddings`
    (the dict returned by load_glove_embeddings) is passed in rather
    than reloaded here, since loading the ~1.2M-line GloVe file takes
    a minute or two -- doing that once outside the fold loop, instead
    of 5 times inside it, saves several minutes.

    Within the 80% training portion of each fold, a further internal
    80/20 split provides the validation set early stopping monitors,
    keeping the fold's held-out test portion completely untouched
    until final prediction.

    `seed` fixes Keras/TensorFlow's own random state (weight
    initialisation, shuffling), which is separate from the
    scikit-learn `random_state` values already fixed elsewhere in this
    module. Two runs with the same `seed` reproduce the same fold
    splits (get_fold_indices always uses RANDOM_SEED) AND the same
    per-fold model initialisation, so results are fully reproducible.
    Varying `seed` across repeated full runs is how genuine
    run-to-run variance in the LSTM's training stability can be
    measured -- see src/evaluation/lstm_stability.py."""
    set_global_seed(seed)
    folds = get_fold_indices(df["sentiment_3class"])
    results = []
    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        inner_train_df, inner_val_df = train_test_split(
            train_df, test_size=0.2,
            stratify=train_df["sentiment_3class"], random_state=RANDOM_SEED,
        )

        tokenizer = build_tokenizer(inner_train_df["lstm_text"])
        X_train = texts_to_padded_sequences(tokenizer, inner_train_df["lstm_text"], max_len=max_len)
        X_val = texts_to_padded_sequences(tokenizer, inner_val_df["lstm_text"], max_len=max_len)
        X_test = texts_to_padded_sequences(tokenizer, test_df["lstm_text"], max_len=max_len)

        y_train = encode_labels(inner_train_df["sentiment_3class"])
        y_val = encode_labels(inner_val_df["sentiment_3class"])

        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = build_embedding_matrix(tokenizer, glove_embeddings, EMBEDDING_DIM)

        model = build_lstm_model(vocab_size, EMBEDDING_DIM, embedding_matrix, max_len=max_len)
        print(f"--- LSTM fold {fold_num}/{len(folds)} ---")
        train_lstm(model, X_train, y_train, X_val, y_val, epochs=epochs)

        preds = model.predict(X_test, verbose=0).argmax(axis=1)
        preds_labels = [INT_TO_LABEL[p] for p in preds]

        results.append({
            "y_true": test_df["sentiment_3class"].tolist(),
            "y_pred": preds_labels,
        })
    return results