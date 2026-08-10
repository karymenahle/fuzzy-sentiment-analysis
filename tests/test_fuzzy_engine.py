"""Unit tests for the Mamdani fuzzy inference engine."""
from src.fuzzy_system.engine import classify_score, infer_sentiment


def test_classify_score_positive_boundary():
    """classify_score's documented breakpoint: Positive starts at 0.10
    (inclusive) -- see engine.py docstring."""
    assert classify_score(0.10) == "Positive"
    assert classify_score(0.50) == "Positive"
    assert classify_score(1.0) == "Positive"


def test_classify_score_negative_boundary():
    """Negative's plateau ends at -0.10 (inclusive)."""
    assert classify_score(-0.10) == "Negative"
    assert classify_score(-0.50) == "Negative"
    assert classify_score(-1.0) == "Negative"


def test_classify_score_neutral_band():
    assert classify_score(0.0) == "Neutral"
    assert classify_score(0.09) == "Neutral"
    assert classify_score(-0.09) == "Neutral"


def test_infer_sentiment_strongly_positive_input():
    """High positive intensity, no negative intensity, and low
    subjectivity (a plain factual-sounding positive statement) should
    produce a positive crisp score."""
    score = infer_sentiment(pos_intensity=0.8, neg_intensity=0.0, subj=0.3)
    assert score > 0


def test_infer_sentiment_strongly_negative_input():
    score = infer_sentiment(pos_intensity=0.0, neg_intensity=0.8, subj=0.3)
    assert score < 0


def test_infer_sentiment_no_evidence_stays_near_neutral():
    """With no positive or negative signal at all, the fuzzy system
    should not invent a strong opinion -- this is the interpretability
    behaviour that distinguishes it from the hard classifiers (see
    demo/app.py's documented "im happy" / no-vocabulary-match case)."""
    score = infer_sentiment(pos_intensity=0.0, neg_intensity=0.0, subj=0.0)
    assert classify_score(score) == "Neutral"


def test_infer_sentiment_output_within_bounds():
    """The crisp output must stay within the [-1, 1] universe of
    discourse regardless of input combination."""
    for pos in (0.0, 0.5, 1.0):
        for neg in (0.0, 0.5, 1.0):
            for subj in (0.0, 0.5, 1.0):
                score = infer_sentiment(pos, neg, subj)
                assert -1.0 <= score <= 1.0


def test_infer_sentiment_is_deterministic():
    """Same inputs must always produce the same output -- unlike the
    LSTM, the fuzzy system has no trained/random component."""
    score_a = infer_sentiment(0.6, 0.2, 0.4)
    score_b = infer_sentiment(0.6, 0.2, 0.4)
    assert score_a == score_b