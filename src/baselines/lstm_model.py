"""
LSTM model architecture and training - 6G7V0007 MSc Project
Karyme Nahle Acosta

Builds a modest LSTM classifier with a frozen GloVe embedding layer.
Kept deliberately small (single LSTM layer, dropout, early stopping)
given the dataset only has 706 posts -- a larger network would overfit
quickly.
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential


def build_lstm_model(vocab_size: int, embedding_dim: int, embedding_matrix,
                      max_len: int, lstm_units: int = 64, dropout_rate: float = 0.3) -> Sequential:
    """Builds and compiles the LSTM model. The Embedding layer is
    initialised with the pre-trained GloVe matrix and frozen
    (trainable=False), so only the LSTM and Dense layers are actually
    learned from the training data."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
        ),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        Dropout(dropout_rate),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_lstm(model, X_train, y_train, X_val, y_val,
                epochs: int = 30, batch_size: int = 16, patience: int = 3,
                use_class_weight: bool = True):
    """Trains with early stopping on validation loss: if val_loss
    doesn't improve for `patience` consecutive epochs, training stops
    and the best-performing weights are restored, instead of continuing
    to overfit.

    `use_class_weight=True` (default) scales each class's contribution
    to the loss inversely to its frequency in y_train, the same
    imbalance mitigation already applied to the SVM baseline
    (class_weight="balanced"). Without it, the LSTM -- like Naive Bayes
    with fit_prior=True -- tends to ignore the minority Neutral class
    almost entirely."""
    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        print("Class weights:", class_weight)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1,
    )
    return history