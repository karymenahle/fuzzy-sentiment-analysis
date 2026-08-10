"""Unit tests for the three preprocessing branches (lexicon/TF-IDF/LSTM)."""
from src.preprocessing.text_cleaning import (
    branch_a_lexicon,
    branch_b_tfidf,
    branch_c_lstm,
    preprocess_pipeline,
)


def test_branch_a_preserves_case_and_punctuation():
    """VADER/TextBlob rely on case and punctuation (e.g. "GREAT!!!" vs
    "great") -- branch_a must not normalise either away."""
    text = "This is AMAZING!!!"
    result = branch_a_lexicon(text)
    assert "AMAZING" in result
    assert "!!!" in result


def test_branch_a_strips_urls_mentions_hashtags():
    text = "Check this out https://example.com @someone #excited"
    result = branch_a_lexicon(text)
    assert "http" not in result
    assert "@someone" not in result
    assert "#" not in result


def test_branch_a_hashtag_word_removed_by_default():
    """remove_hashtag_word=True is the default, and is the leakage
    mitigation documented in text_cleaning.py: many posts have their
    own sentiment label embedded as a hashtag (e.g. "#Joy"), so the
    whole hashtag -- symbol AND word -- is stripped, not just the
    symbol."""
    result = branch_a_lexicon("So #excited for this")
    assert "excited" not in result


def test_branch_a_can_keep_hashtag_word():
    """With remove_hashtag_word=False, only the # symbol is stripped
    and the word itself is kept."""
    result = branch_a_lexicon("So #excited for this", remove_hashtag_word=False)
    assert "excited" in result


def test_branch_b_lowercases_and_removes_stopwords():
    result = branch_b_tfidf("I AM so Happy about this!")
    assert result["text"] == result["text"].lower()
    assert "i" not in result["tokens"]
    assert "am" not in result["tokens"]
    assert "happy" in result["tokens"]


def test_branch_b_lemmatises_tokens():
    result = branch_b_tfidf("I am running and jumping happily")
    # lemmatisation should reduce inflected forms to their base form
    assert "running" not in result["tokens"] or "run" in result["tokens"]


def test_branch_b_expands_contractions_before_tokenising():
    """"don't" and "do not" must be treated identically once expanded
    -- both should produce the same remaining content word, "like"
    (the contraction word "not" is itself an NLTK stopword and is
    removed by branch_b's full stopword removal, unlike branch_c which
    deliberately keeps it -- see test_branch_c_preserves_stopwords_for_negation)."""
    result_with = branch_b_tfidf("I don't like this")
    result_without_apostrophe = branch_b_tfidf("I do not like this")
    assert result_with["tokens"] == result_without_apostrophe["tokens"]
    assert "like" in result_with["tokens"]


def test_branch_c_preserves_stopwords_for_negation():
    """The LSTM branch must NOT remove stopwords like "not", since
    removing them would destroy negation, which the LSTM relies on
    sequential context to interpret (see text_cleaning.py docstring)."""
    result = branch_c_lstm("I do not like this")
    assert "not" in result["tokens"]


def test_branch_c_does_not_lemmatise():
    """Preserves the exact surface form so it matches GloVe's
    vocabulary as closely as possible (see text_cleaning.py docstring)."""
    result = branch_c_lstm("I am running happily")
    assert "running" in result["tokens"]


def test_preprocess_pipeline_returns_all_three_representations():
    result = preprocess_pipeline("I am so happy today!")
    assert "lexicon_text" in result
    assert "tfidf_text" in result
    assert "tfidf_tokens" in result
    assert "lstm_text" in result
    assert "lstm_tokens" in result


def test_preprocess_pipeline_handles_empty_string():
    """An empty or purely non-alphabetic post (e.g. just an emoji or a
    URL) must not raise -- it should just produce empty tokens."""
    result = preprocess_pipeline("https://example.com")
    assert result["tfidf_tokens"] == []
    assert result["lstm_tokens"] == []