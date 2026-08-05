"""
TF-IDF vectorization - 6G7V0007 MSc Project
Karyme Nahle Acosta

Builds the shared TF-IDF representation used by both the Naive Bayes and
SVM baselines, from the `tfidf_text` column produced by the
preprocessing pipeline (lowercased, lemmatised, stopwords removed).
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Returns an unfitted TfidfVectorizer. `max_features` caps the
    vocabulary size to the most frequent terms, which matters here since
    the dataset is small (706 posts) and a huge sparse vocabulary would
    make the classifiers prone to overfitting."""
    return TfidfVectorizer(max_features=max_features)