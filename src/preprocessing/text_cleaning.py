"""
Text preprocessing pipeline - 6G7V0007 MSc Project
Karyme Nahle Acosta

A minimal shared cleaning step (URLs, mentions, hashtags) is applied
first, followed by three representation-specific branches, because text
normalisation requirements are in tension across systems:

  Branch A (lexicon) -> VADER / TextBlob for the fuzzy system.
                        Preserves case, punctuation and emoticons, since
                        VADER uses those cues to modulate intensity.
  Branch B (tfidf)   -> Naive Bayes / SVM.
                        Full normalisation: lowercase, no punctuation,
                        no stopwords, POS-aware lemmatisation.
  Branch C (lstm)    -> LSTM model with GloVe embeddings.
                        Lowercase and tokenised, but stopwords are NOT
                        removed (preserves negation words such as "not")
                        and lemmatisation is NOT applied (preserves GloVe
                        vocabulary coverage).

`remove_hashtag_word` is applied to ALL THREE branches equally: many
posts have the original sentiment label embedded literally as a hashtag
(e.g. "#Joy"), which leaks the target variable into the input text for
the classifiers and the fuzzy system alike if not removed.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Downloads (if missing) the NLTK resources this module needs.
    Called immediately below, so importing this module is enough to
    guarantee the resources are available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except (LookupError, OSError):
            nltk.download(pkg, quiet=True)


ensure_nltk_resources()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def _wordnet_pos(tag: str) -> str:
    """Converts a Penn Treebank POS tag to WordNet's format, so that
    lemmatisation correctly handles verbs/adjectives/adverbs instead of
    assuming everything is a noun."""
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def _strip_urls_mentions_hashtags(text: str, remove_hashtag_word: bool) -> str:
    """Step shared by all three branches: always removes URLs and
    mentions; for hashtags, removes the whole word if
    remove_hashtag_word=True (leakage mitigation), or only the # symbol
    if False."""
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if remove_hashtag_word:
        text = HASHTAG_RE.sub(" ", text)
    else:
        text = HASHTAG_RE.sub(lambda m: m.group(0)[1:], text)
    return MULTI_SPACE_RE.sub(" ", text).strip()


# Common English contractions and social-media shorthand, expanded before
# tokenisation in Branches B and C. This avoids two problems at once:
# dropping meaningful single-letter shorthand (e.g. "u" -> "you") if a
# length filter is applied later, and leaving orphaned fragments such as
# "n't" or "'s" in the token list once the apostrophe splits the word.
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "can't": "cannot",
    "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "it's": "it is",
    "that's": "that is", "there's": "there is", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "let's": "let us",
    "u": "you", "ur": "your", "r": "are", "im": "i am",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "lol": "laughing out loud", "omg": "oh my god", "btw": "by the way",
}
CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b"
)


def _expand_contractions(text: str) -> str:
    """Replaces known contractions/shorthand with their expanded form.
    Must run on already-lowercased text, since the dictionary keys are
    lowercase."""
    return CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)


def branch_a_lexicon(text: str, remove_hashtag_word: bool = True) -> str:
    """Branch A: for VADER/TextBlob. Preserves case and punctuation."""
    return _strip_urls_mentions_hashtags(text, remove_hashtag_word)


def branch_b_tfidf(text: str, remove_hashtag_word: bool = True) -> dict:
    """Branch B: for Naive Bayes / SVM. Full normalisation + POS-aware
    lemmatisation."""
    cleaned = _strip_urls_mentions_hashtags(text, remove_hashtag_word)
    cleaned = cleaned.lower()
    cleaned = _expand_contractions(cleaned)
    cleaned = NON_ALPHA_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()

    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tagged = pos_tag(tokens)
    lemmas = [LEMMATIZER.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]

    return {"tokens": lemmas, "text": " ".join(lemmas)}


def branch_c_lstm(text: str, remove_hashtag_word: bool = True) -> dict:
    """Branch C: for the LSTM. Lowercase + tokenised, without removing
    stopwords (preserves negation) and without lemmatising (preserves
    GloVe vocabulary)."""
    cleaned = _strip_urls_mentions_hashtags(text, remove_hashtag_word)
    cleaned = cleaned.lower()
    cleaned = _expand_contractions(cleaned)
    cleaned = NON_ALPHA_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()

    tokens = word_tokenize(cleaned)
    return {"tokens": tokens, "text": " ".join(tokens)}


def preprocess_pipeline(text: str, remove_hashtag_word: bool = True) -> dict:
    """Applies all three branches to the same text and returns every
    resulting representation in a single dictionary."""
    lexicon_text = branch_a_lexicon(text, remove_hashtag_word)
    tfidf_out = branch_b_tfidf(text, remove_hashtag_word)
    lstm_out = branch_c_lstm(text, remove_hashtag_word)

    return {
        "lexicon_text": lexicon_text,
        "tfidf_tokens": tfidf_out["tokens"],
        "tfidf_text": tfidf_out["text"],
        "lstm_tokens": lstm_out["tokens"],
        "lstm_text": lstm_out["text"],
    }