"""
Fuzzy linguistic variables, membership functions and rule base -
6G7V0007 MSc Project
Karyme Nahle Acosta

Defines the 3 input variables (positive_intensity, negative_intensity,
subjectivity), each over [0, 1] with 3 terms (Low, Medium, High), and the
output variable (sentiment_score) over [-1, 1] with 3 terms (Negative,
Neutral, Positive).

Trapezoidal shapes are used for the outer terms of each variable (Low,
High), so that values beyond a certain point get full membership instead
of tailing off indefinitely. Triangular shapes are used for the middle
term (Medium). Breakpoints give each term a roughly proportionate share
of the universe with moderate overlap between adjacent terms, which is
what produces the smooth, graded transitions that distinguish a fuzzy
system from a hard-boundary classifier. These are an initial, principled
design rather than data-tuned values.

The rule base follows Mamdani-style inference: antecedents combined with
the minimum (AND) operator, rule outputs aggregated with the maximum
operator, and the aggregated fuzzy set defuzzified with the centroid
method -- the default configuration provided by scikit-fuzzy.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Input variables: positive_intensity, negative_intensity ---
_intensity_universe = np.arange(0, 1.001, 0.001)

positive_intensity = ctrl.Antecedent(_intensity_universe, "positive_intensity")
positive_intensity["Low"] = fuzz.trapmf(_intensity_universe, [0, 0, 0.15, 0.35])
positive_intensity["Medium"] = fuzz.trimf(_intensity_universe, [0.20, 0.40, 0.60])
positive_intensity["High"] = fuzz.trapmf(_intensity_universe, [0.45, 0.65, 1.0, 1.0])

negative_intensity = ctrl.Antecedent(_intensity_universe, "negative_intensity")
negative_intensity["Low"] = fuzz.trapmf(_intensity_universe, [0, 0, 0.15, 0.35])
negative_intensity["Medium"] = fuzz.trimf(_intensity_universe, [0.20, 0.40, 0.60])
negative_intensity["High"] = fuzz.trapmf(_intensity_universe, [0.45, 0.65, 1.0, 1.0])

# --- Input variable: subjectivity ---
_subjectivity_universe = np.arange(0, 1.001, 0.001)

subjectivity = ctrl.Antecedent(_subjectivity_universe, "subjectivity")
subjectivity["Low"] = fuzz.trapmf(_subjectivity_universe, [0, 0, 0.20, 0.45])
subjectivity["Medium"] = fuzz.trimf(_subjectivity_universe, [0.30, 0.50, 0.70])
subjectivity["High"] = fuzz.trapmf(_subjectivity_universe, [0.55, 0.80, 1.0, 1.0])

# --- Output variable: sentiment_score ---
_score_universe = np.arange(-1, 1.001, 0.001)

sentiment_score = ctrl.Consequent(_score_universe, "sentiment_score")
sentiment_score["Negative"] = fuzz.trapmf(_score_universe, [-1.0, -1.0, -0.5, -0.10])
sentiment_score["Neutral"] = fuzz.trimf(_score_universe, [-0.25, 0, 0.25])
sentiment_score["Positive"] = fuzz.trapmf(_score_universe, [0.10, 0.5, 1.0, 1.0])


# --- Rule base (Mamdani inference, 11 rules) ---
rules = [
    ctrl.Rule(positive_intensity["High"] & negative_intensity["Low"], sentiment_score["Positive"]),      # R1
    ctrl.Rule(positive_intensity["High"] & negative_intensity["Medium"], sentiment_score["Positive"]),   # R2
    ctrl.Rule(positive_intensity["Medium"] & negative_intensity["Low"], sentiment_score["Positive"]),    # R3
    ctrl.Rule(positive_intensity["Low"] & negative_intensity["High"], sentiment_score["Negative"]),      # R4
    ctrl.Rule(positive_intensity["Medium"] & negative_intensity["High"], sentiment_score["Negative"]),   # R5
    ctrl.Rule(positive_intensity["Low"] & negative_intensity["Medium"], sentiment_score["Negative"]),    # R6
    ctrl.Rule(positive_intensity["High"] & negative_intensity["High"], sentiment_score["Neutral"]),      # R7 - conflicting
    ctrl.Rule(positive_intensity["Medium"] & negative_intensity["Medium"], sentiment_score["Neutral"]),  # R8 - balanced
    ctrl.Rule(
        positive_intensity["Low"] & negative_intensity["Low"] & subjectivity["Low"],
        sentiment_score["Neutral"],
    ),  # R9 - objective
    ctrl.Rule(
        positive_intensity["Low"] & negative_intensity["Low"] & subjectivity["Medium"],
        sentiment_score["Neutral"],
    ),  # R10
    ctrl.Rule(
        positive_intensity["Low"] & negative_intensity["Low"] & subjectivity["High"],
        sentiment_score["Neutral"],
    ),  # R11 - borderline
]

_control_system = ctrl.ControlSystem(rules)


def infer_sentiment(pos_intensity: float, neg_intensity: float, subj: float) -> float:
    """Runs Mamdani inference (min AND, max aggregation, centroid
    defuzzification -- scikit-fuzzy's default) and returns the crisp
    sentiment_score in [-1, 1]."""
    simulation = ctrl.ControlSystemSimulation(_control_system)
    simulation.input["positive_intensity"] = pos_intensity
    simulation.input["negative_intensity"] = neg_intensity
    simulation.input["subjectivity"] = subj
    simulation.compute()
    return simulation.output["sentiment_score"]


def classify_score(score: float) -> str:
    """Maps the continuous sentiment_score to one of the 3 classes used
    everywhere else in the project, following the same breakpoints as
    the output membership functions (Positive starts at 0.10, Negative's
    plateau ends at -0.10)."""
    if score >= 0.10:
        return "Positive"
    if score <= -0.10:
        return "Negative"
    return "Neutral"