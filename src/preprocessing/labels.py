"""
Mapping of the 191 granular labels in the original `Sentiment` column to
the 3 classes (Positive / Negative / Neutral). 
"""

POSITIVE_LABELS = {
    "positive", "positivity", "joy", "happiness", "happy", "excitement",
    "contentment", "gratitude", "curiosity", "serenity", "hopeful", "hope",
    "acceptance", "pride", "elation", "euphoria", "enthusiasm",
    "determination", "admiration", "adoration", "amazement", "amusement",
    "anticipation", "appreciation", "arousal", "awe", "blessed", "calmness",
    "captivation", "celebration", "charm", "confidence", "confident",
    "connection", "contemplation", "coziness", "creativity",
    "creative inspiration", "culinary adventure", "culinaryodyssey",
    "elegance", "empowerment", "enchantment", "energy", "engagement",
    "enjoyment", "fulfillment", "grandeur", "grateful", "harmony",
    "heartwarming", "iconic", "imagination", "inspiration", "inspired",
    "intrigue", "journey", "joy in baking", "joyfulreunion", "kind",
    "kindness", "love", "marvel", "mesmerizing", "mindfulness",
    "motivation", "optimism", "overjoyed", "playful", "playfuljoy",
    "radiance", "relief", "renewed effort", "reverence", "romance",
    "satisfaction", "spark", "success", "thrill", "thrilling journey",
    "touched", "tranquility", "triumph", "vibrancy", "whimsy", "wonder",
    "wonderment", "zest", "freedom", "free-spirited", "friendship",
    "breakthrough", "accomplishment", "adventure", "affection", "colorful",
    "dazzle", "empathetic", "compassion", "compassionate",
    "nature's beauty", "ocean's freedom", "winter magic", "festivejoy",
    "solace", "resilience", "reflection", "rejuvenation", "tenderness",
    "sympathy", "celestial wonder", "envisioning history", "exploration",
    "immersion", "hypnotic", "melodic", "proud", "dreamchaser",
    "innerjourney", "artisticburst", "emotionalstorm",
    "runway creativity", "whispers of the past", "good",
    "adrenaline", "ecstasy",
}

NEGATIVE_LABELS = {
    "negative", "sad", "sadness", "sorrow", "despair", "grief", "anger",
    "fear", "fearful", "frustration", "frustrated", "disgust",
    "disappointed", "disappointment", "hate", "heartbreak", "heartache",
    "jealous", "jealousy", "loneliness", "loss", "lostlove", "betrayal",
    "bitterness", "bitter", "boredom", "darkness",
    "desolation", "desperation", "devastated", "embarrassed", "envy",
    "envious", "exhaustion", "helplessness", "isolation", "melancholy",
    "obstacle", "regret", "resentment", "shame", "suffering", "suspense",
    "anxiety", "apprehensive", "intimidation", "miscalculation",
    "overwhelmed", "pressure", "ruins", "solitude", "yearning",
    "dismissive", "bad", "challenge", "numbness",
}

NEUTRAL_LABELS = {
    "neutral", "indifference", "ambivalence", "pensive", "confusion",
    "surprise", "emotion", "mischievous", "nostalgia", "bittersweet",
}


def map_sentiment(label: str) -> str:
    """Converts a granular label into Positive / Negative / Neutral.

    Returns "Unmapped" if the label does not appear in any of the three
    sets, so new/unseen cases can be caught and reviewed explicitly
    instead of silently defaulting to a class.
    """
    key = str(label).strip().lower()
    if key in POSITIVE_LABELS:
        return "Positive"
    if key in NEGATIVE_LABELS:
        return "Negative"
    if key in NEUTRAL_LABELS:
        return "Neutral"
    return "Unmapped"