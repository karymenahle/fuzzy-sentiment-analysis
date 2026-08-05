"""
GloVe embedding loading - 6G7V0007 MSc Project
Karyme Nahle Acosta

Loads pre-trained GloVe vectors (glove.twitter.27B.100d.txt, chosen for
domain match with the informal, short-form register of the dataset) and
builds an embedding matrix restricted to the project's own vocabulary,
for use as a frozen Embedding layer in the LSTM.
"""
import numpy as np


def load_glove_embeddings(path) -> dict:
    """Reads a GloVe text file into a dict {word: vector}. Lines that
    don't parse as "word followed by N floats" are skipped and counted,
    since GloVe's Twitter release occasionally has malformed lines."""
    embeddings = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                vector = np.asarray(parts[1:], dtype="float32")
            except ValueError:
                skipped += 1
                continue
            embeddings[word] = vector
    if skipped:
        print(f"GloVe: skipped {skipped} malformed lines while loading.")
    return embeddings


def build_embedding_matrix(tokenizer, glove_embeddings: dict, embedding_dim: int) -> np.ndarray:
    """Builds a (vocab_size, embedding_dim) matrix aligned with the
    tokenizer's word index: row i holds the GloVe vector for the word
    whose integer index is i. Words not found in GloVe (including the
    padding index 0 and the <OOV> token) are left as all-zero rows.

    Also prints vocabulary coverage, since a low coverage percentage
    would mean the LSTM is starting with little useful information for
    a large share of its vocabulary."""
    vocab_size = len(tokenizer.word_index) + 1
    matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    for word, idx in tokenizer.word_index.items():
        vector = glove_embeddings.get(word)
        if vector is not None:
            matrix[idx] = vector
            found += 1

    coverage = found / (vocab_size - 1) * 100
    print(f"GloVe coverage: {found}/{vocab_size - 1} words found ({coverage:.1f}%).")
    return matrix