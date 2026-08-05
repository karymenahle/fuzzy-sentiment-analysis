"""
Naive Bayes baseline - 6G7V0007 MSc Project
Karyme Nahle Acosta

Multinomial Naive Bayes over the TF-IDF representation. `fit_prior`
defaults to False: comparing both variants (see project notes) showed
this outperforms the empirical-prior default on the minority Neutral
class, and on macro F1 and accuracy overall, so it is the setting used
for the project's actual results. The parameter is kept so both variants
remain reproducible if needed again.
"""
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X_train, y_train, fit_prior: bool = False) -> MultinomialNB:
    """Trains a MultinomialNB classifier on already-vectorized TF-IDF
    features."""
    model = MultinomialNB(fit_prior=fit_prior)
    model.fit(X_train, y_train)
    return model