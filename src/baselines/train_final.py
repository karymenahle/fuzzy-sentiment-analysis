"""
Final model training orchestrator (for the interactive demo).
Usage: python -m src.baselines.train_final

Trains Naive Bayes, SVM, and the LSTM on the FULL preprocessed dataset
(no train/test split, no cross-validation) and saves each one's
artefacts to models/ via src/baselines/persistence.py.

This is deliberately separate from cross_validate_* (used for reporting
metrics in the dissertation): those functions hold out a test fold on
purpose, so their models must never be the ones shipped in the demo. A
demo has no metrics to protect by holding out data -- it should use
every available post to make its predictions as good as possible.

The LSTM still uses an internal 80/20 split purely to give early
stopping a validation set to monitor; that split does not appear in
any reported metric.
"""
from sklearn.model_selection import train_test_split

from src.baselines.glove_embeddings import build_embedding_matrix, load_glove_embeddings
from src.baselines.lstm_data import build_tokenizer, encode_labels, texts_to_padded_sequences
from src.baselines.lstm_model import build_lstm_model, train_lstm
from src.baselines.naive_bayes import train_naive_bayes
from src.baselines.persistence import save_lstm, save_naive_bayes, save_svm
from src.baselines.run_all import load_preprocessed
from src.baselines.svm import train_svm
from src.baselines.vectorizer import build_tfidf_vectorizer
from src.utils.config import DATA_EMBEDDINGS_DIR, EMBEDDING_DIM, GLOVE_FILENAME, RANDOM_SEED
from src.utils.seed import set_global_seed


def train_and_save_naive_bayes(df) -> None:
    print("\nTraining final Naive Bayes model on the full dataset...")
    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(df["tfidf_text"])
    model = train_naive_bayes(X, df["sentiment_3class"])
    save_naive_bayes(model, vectorizer)
    print("Saved: models/naive_bayes_model.joblib, models/naive_bayes_vectorizer.joblib")


def train_and_save_svm(df) -> None:
    print("\nTraining final SVM model on the full dataset...")
    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(df["tfidf_text"])
    model = train_svm(X, df["sentiment_3class"])
    save_svm(model, vectorizer)
    print("Saved: models/svm_model.joblib, models/svm_vectorizer.joblib")


def train_and_save_lstm(df, max_len: int = 30) -> None:
    print("\nLoading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings(DATA_EMBEDDINGS_DIR / GLOVE_FILENAME)
    print(f"Loaded {len(glove_embeddings)} GloVe vectors.")

    print("\nTraining final LSTM model on the full dataset...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["sentiment_3class"], random_state=RANDOM_SEED,
    )

    tokenizer = build_tokenizer(train_df["lstm_text"])
    X_train = texts_to_padded_sequences(tokenizer, train_df["lstm_text"], max_len=max_len)
    X_val = texts_to_padded_sequences(tokenizer, val_df["lstm_text"], max_len=max_len)
    y_train = encode_labels(train_df["sentiment_3class"])
    y_val = encode_labels(val_df["sentiment_3class"])

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = build_embedding_matrix(tokenizer, glove_embeddings, EMBEDDING_DIM)

    model = build_lstm_model(vocab_size, EMBEDDING_DIM, embedding_matrix, max_len=max_len)
    train_lstm(model, X_train, y_train, X_val, y_val)

    save_lstm(model, tokenizer)
    print("Saved: models/lstm_model.keras, models/lstm_tokenizer.joblib")


def run() -> None:
    set_global_seed(RANDOM_SEED)
    df = load_preprocessed()
    print(f"Loaded {len(df)} preprocessed posts.")

    train_and_save_naive_bayes(df)
    train_and_save_svm(df)
    train_and_save_lstm(df)

    print("\nAll final models trained and saved to models/.")
    print("The fuzzy system needs no saved artefacts -- its rules are fixed by design.")


if __name__ == "__main__":
    run()