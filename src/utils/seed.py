"""
Global random seed control - 6G7V0007 MSc Project
Karyme Nahle Acosta

Keras/TensorFlow weight initialisation draws from its own global random
state, separate from the `random_state` parameters already fixed on the
scikit-learn side (StratifiedKFold, train_test_split, SVM). Without
seeding it explicitly, two runs of the exact same LSTM training code can
still initialise different weights and diverge -- which is what produced
the fold 4 instability observed when comparing two separate runs of
cross_validate_lstm.
"""
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Seeds Python's random module, NumPy, and TensorFlow, covering the
    three sources of randomness Keras model initialisation and training
    can draw from."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)