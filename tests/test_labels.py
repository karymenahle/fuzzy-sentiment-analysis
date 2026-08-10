"""Unit tests for the 191-label -> 3-class sentiment mapping."""
from src.preprocessing.labels import map_sentiment


def test_known_positive_label():
    assert map_sentiment("joy") == "Positive"


def test_known_negative_label():
    assert map_sentiment("sadness") == "Negative"


def test_known_neutral_label():
    assert map_sentiment("neutral") == "Neutral"


def test_unmapped_label_returns_unmapped():
    """A label that is not in any of the three sets must not silently
    fall into a default class -- see map_sentiment's docstring."""
    assert map_sentiment("this is not a real label") == "Unmapped"


def test_mapping_is_case_insensitive():
    assert map_sentiment("Joy") == "Positive"
    assert map_sentiment("JOY") == "Positive"
    assert map_sentiment("SaDnEsS") == "Negative"


def test_mapping_strips_surrounding_whitespace():
    assert map_sentiment("  joy  ") == "Positive"
    assert map_sentiment("\tneutral\n") == "Neutral"


def test_ambiguous_borderline_labels_documented_as_positive():
    """These were explicitly documented in the project's own labelling
    decisions as ambiguous cases resolved to Positive -- this test
    guards against an accidental change to that documented decision."""
    assert map_sentiment("acceptance") == "Positive"
    assert map_sentiment("mindfulness") == "Positive"


def test_no_label_appears_in_more_than_one_set():
    """A label present in two of the three sets would make map_sentiment
    depend on set iteration/check order rather than a clear mapping --
    this guards against that ever silently happening as the label sets
    are edited."""
    from src.preprocessing.labels import NEGATIVE_LABELS, NEUTRAL_LABELS, POSITIVE_LABELS

    assert POSITIVE_LABELS.isdisjoint(NEGATIVE_LABELS)
    assert POSITIVE_LABELS.isdisjoint(NEUTRAL_LABELS)
    assert NEGATIVE_LABELS.isdisjoint(NEUTRAL_LABELS)