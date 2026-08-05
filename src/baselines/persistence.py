"""
Model persistence for the interactive demo - 6G7V0007 MSc Project
Karyme Nahle Acosta

Saves/loads the artefacts each system needs to make a prediction on a
single new post, without retraining:
  - Naive Bayes / SVM: the fitted TfidfVectorizer + the fitted classifier
    (both needed -- a classifier alone cannot vectorize new raw text).
  - LSTM: the fitted Keras Tokenizer + the trained model (the tokenizer
    maps new text to the same integer vocabulary the model was trained
    on; without it the model's input would be meaningless).
  - Fuzzy: nothing to save -- its rules and membership functions are
    fixed by design (see fuzzy_system/engine.py), so it is always ready.

These are FINAL models trained on the full dataset (see
src/baselines/train_final.py), not the per-fold models used during
cross-validation -- a demo has no held-out test set to protect.
"""
import joblib

from src.utils.config import MODELS_DIR


def save_naive_bayes(model, vectorizer) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "naive_bayes_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "naive_bayes_vectorizer.joblib")


def load_naive_bayes():
    model = joblib.load(MODELS_DIR / "naive_bayes_model.joblib")
    vectorizer = joblib.load(MODELS_DIR / "naive_bayes_vectorizer.joblib")
    return model, vectorizer


def save_svm(model, vectorizer) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "svm_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "svm_vectorizer.joblib")


def load_svm():
    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    vectorizer = joblib.load(MODELS_DIR / "svm_vectorizer.joblib")
    return model, vectorizer


def save_lstm(model, tokenizer) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_model.keras")
    joblib.dump(tokenizer, MODELS_DIR / "lstm_tokenizer.joblib")


def load_lstm():
    from tensorflow.keras.models import load_model  # local import: avoids
    # paying TensorFlow's import cost for code paths (NB/SVM-only) that
    # never touch the LSTM.
    model = load_model(MODELS_DIR / "lstm_model.keras")
    tokenizer = joblib.load(MODELS_DIR / "lstm_tokenizer.joblib")
    return model, tokenizer