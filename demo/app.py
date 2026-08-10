"""
Streamlit demo - 6G7V0007 MSc Project
Karyme Nahle Acosta

Side-by-side comparison of the fuzzy system, Naive Bayes, SVM, and LSTM
on a single post the user types in. Uses the FINAL models trained on
100% of the dataset (models/, produced by src/baselines/train_final.py)
-- never the per-fold cross-validation models, which exist only to
report metrics.

Run with:
    streamlit run demo/app.py

Known limitation (documented for the dissertation's limitations
chapter): the TF-IDF vocabulary used by Naive Bayes and SVM comes from
only 706 posts, so everyday words the dataset happens not to use
(e.g. "happy", "wonderful") are out-of-vocabulary and contribute
nothing to those two models' predictions -- the demo flags this
directly by showing which of the typed words were actually recognised.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.persistence import load_lstm, load_naive_bayes, load_svm
from src.baselines.lstm_data import INT_TO_LABEL, texts_to_padded_sequences
from src.fuzzy_system.engine import classify_score, infer_sentiment
from src.fuzzy_system.inputs import compute_fuzzy_inputs
from src.preprocessing.text_cleaning import preprocess_pipeline

LABEL_COLORS = {"Positive": "green", "Negative": "red", "Neutral": "gray"}


@st.cache_resource
def get_models():
    """Loads all final models once per session, not on every keystroke."""
    nb_model, nb_vectorizer = load_naive_bayes()
    svm_model, svm_vectorizer = load_svm()
    lstm_model, lstm_tokenizer = load_lstm()
    return {
        "nb": (nb_model, nb_vectorizer),
        "svm": (svm_model, svm_vectorizer),
        "lstm": (lstm_model, lstm_tokenizer),
    }


def run_fuzzy(lexicon_text: str) -> dict:
    inputs = compute_fuzzy_inputs(lexicon_text)
    score = infer_sentiment(inputs["positive_intensity"], inputs["negative_intensity"], inputs["subjectivity"])
    label = classify_score(score)
    return {"label": label, "score": score, "inputs": inputs}


def run_naive_bayes(tfidf_text: str, tfidf_tokens: list, models: dict) -> dict:
    model, vectorizer = models["nb"]
    X = vectorizer.transform([tfidf_text])
    label = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))

    vocab = vectorizer.vocabulary_
    recognised = [t for t in tfidf_tokens if t in vocab]
    return {"label": label, "proba": proba, "recognised_tokens": recognised}


def run_svm(tfidf_text: str, tfidf_tokens: list, models: dict) -> dict:
    model, vectorizer = models["svm"]
    X = vectorizer.transform([tfidf_text])
    label = model.predict(X)[0]

    vocab = vectorizer.vocabulary_
    recognised = [t for t in tfidf_tokens if t in vocab]
    return {"label": label, "recognised_tokens": recognised}


def run_lstm(lstm_text: str, models: dict, max_len: int = 30) -> dict:
    model, tokenizer = models["lstm"]
    X = texts_to_padded_sequences(tokenizer, [lstm_text], max_len=max_len)
    probs = model.predict(X, verbose=0)[0]
    pred_idx = probs.argmax()
    label = INT_TO_LABEL[pred_idx]
    proba = {INT_TO_LABEL[i]: float(p) for i, p in enumerate(probs)}
    return {"label": label, "proba": proba}


def render_label(label: str) -> None:
    color = LABEL_COLORS.get(label, "gray")
    st.markdown(f"### :{color}[{label}]")


def render_proba(proba: dict) -> None:
    for cls, p in sorted(proba.items(), key=lambda kv: -kv[1]):
        st.progress(float(p), text=f"{cls}: {p:.1%}")


def main() -> None:
    st.set_page_config(page_title="Fuzzy Sentiment Analysis Demo", layout="wide")
    st.title("Fuzzy Logic vs. Naive Bayes vs. SVM vs. LSTM")
    st.caption(
        "Modelling Linguistic Variables and Fuzzy Rules for Social Media Sentiment "
        "Analysis -- MSc dissertation demo (6G7V0007)"
    )

    models = get_models()

    text = st.text_area(
        "Type a social media post",
        placeholder="e.g. I embrace the joy and acceptance of this beautiful moment",
        height=100,
    )

    if not text.strip():
        st.info("Type a post above to see all four systems' predictions side by side.")
        return

    processed = preprocess_pipeline(text)

    col_fuzzy, col_nb, col_svm, col_lstm = st.columns(4)

    with col_fuzzy:
        st.subheader("Fuzzy System")
        result = run_fuzzy(processed["lexicon_text"])
        render_label(result["label"])
        st.metric("Sentiment score", f"{result['score']:.3f}", help="Crisp output in [-1, 1] after centroid defuzzification")
        with st.expander("Fuzzy inputs (interpretability)"):
            st.write(f"Positive intensity (VADER pos): {result['inputs']['positive_intensity']:.3f}")
            st.write(f"Negative intensity (VADER neg): {result['inputs']['negative_intensity']:.3f}")
            st.write(f"Subjectivity (TextBlob): {result['inputs']['subjectivity']:.3f}")

    with col_nb:
        st.subheader("Naive Bayes")
        result = run_naive_bayes(processed["tfidf_text"], processed["tfidf_tokens"], models)
        render_label(result["label"])
        render_proba(result["proba"])
        _render_vocab_note(processed["tfidf_tokens"], result["recognised_tokens"])

    with col_svm:
        st.subheader("SVM")
        result = run_svm(processed["tfidf_text"], processed["tfidf_tokens"], models)
        render_label(result["label"])
        st.caption("No confidence score shown: this SVM was trained without probability=True.")
        _render_vocab_note(processed["tfidf_tokens"], result["recognised_tokens"])

    with col_lstm:
        st.subheader("LSTM")
        result = run_lstm(processed["lstm_text"], models)
        render_label(result["label"])
        render_proba(result["proba"])


def _render_vocab_note(all_tokens: list, recognised_tokens: list) -> None:
    """Flags the known TF-IDF out-of-vocabulary limitation directly in
    the demo, rather than letting an unrecognised word silently produce
    a misleading prediction."""
    unrecognised = [t for t in all_tokens if t not in recognised_tokens]
    if unrecognised:
        st.caption(f"⚠️ Not in vocabulary: {', '.join(unrecognised)}")


if __name__ == "__main__":
    main()
