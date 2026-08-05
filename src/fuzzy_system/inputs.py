"""
Fuzzy input variables - 6G7V0007 MSc Project
Karyme Nahle Acosta

Computes the three linguistic variable inputs the fuzzy system uses:
  - positive_intensity : from VADER's `pos` score
  - negative_intensity : from VADER's `neg` score
  - subjectivity        : from TextBlob, independently of VADER

All three are on the [0, 1] scale, matching the universe of discourse
used by the membership functions in engine.py.

Uses the "lexicon" branch of the text-cleaning pipeline as input, since
VADER and TextBlob rely on case, punctuation and emoticons that the
other two branches deliberately strip out.
"""
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_VADER = SentimentIntensityAnalyzer()


def compute_fuzzy_inputs(lexicon_text: str) -> dict:
    """Takes text prepared by branch_a_lexicon() and returns the 3
    fuzzy input values, plus the raw VADER scores for inspection."""
    vader_scores = _VADER.polarity_scores(lexicon_text)
    subjectivity = TextBlob(lexicon_text).sentiment.subjectivity

    return {
        "positive_intensity": vader_scores["pos"],
        "negative_intensity": vader_scores["neg"],
        "subjectivity": subjectivity,
        "vader_compound": vader_scores["compound"],
        "vader_neu": vader_scores["neu"],
    }