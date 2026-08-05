"""
SVM baseline - 6G7V0007 MSc Project
Karyme Nahle Acosta

Linear-kernel SVM over the TF-IDF representation. `class_weight`
defaults to "balanced": comparing both variants (see project notes)
showed this outperforms the unweighted default across all three classes,
not only on the minority Neutral class, so it is the setting used for
the project's actual results. The parameter is kept so both variants
remain reproducible if needed again.
"""
from sklearn.svm import SVC


def train_svm(X_train, y_train, class_weight="balanced") -> SVC:
    """Trains a linear-kernel SVM classifier on already-vectorized
    TF-IDF features."""
    model = SVC(kernel="linear", class_weight=class_weight, random_state=42)
    model.fit(X_train, y_train)
    return model