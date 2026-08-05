"""
LSTM sequence preparation - 6G7V0007 MSc Project
Karyme Nahle Acosta

Converts the `lstm_text` column (lowercased, tokenised, but without
stopword removal or lemmatisation -- see text_cleaning.py) into
fixed-length integer sequences the LSTM can consume, and encodes the
3-class labels as integers.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

LABEL_TO_INT = {"Negative": 0, "Neutral": 1, "Positive": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def build_tokenizer(texts, max_words: int = 5000) -> Tokenizer:
    """Fits a Keras Tokenizer on the training texts only. `max_words`
    caps the vocabulary to the most frequent words -- kept modest given
    the dataset only has 706 short posts."""
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer: Tokenizer, texts, max_len: int = 30) -> np.ndarray:
    """Converts texts to integer sequences and pads/truncates them to
    `max_len`. Padding is added at the end ('post'), so real tokens stay
    at the start of the sequence."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def encode_labels(labels) -> np.ndarray:
    """Converts the string class labels to integers (Negative=0,
    Neutral=1, Positive=2), matching the order the model's output layer
    will use."""
    return np.array([LABEL_TO_INT[label] for label in labels])