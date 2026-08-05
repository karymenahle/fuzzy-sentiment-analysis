"""
Preprocessing orchestrator.
Usage: python -m src.preprocessing.run

Steps:
  1. Load the raw CSV and normalise column names/whitespace.
  2. Drop posts with duplicate text.
  3. Map the granular Sentiment labels to 3 classes.
  4. Apply the 3 text-cleaning branches (lexicon / tfidf / lstm).
  5. Save the result to data/processed/.
"""
import pandas as pd

from src.preprocessing.labels import map_sentiment
from src.preprocessing.text_cleaning import preprocess_pipeline
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, RAW_DATASET_FILENAME


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW_DIR / RAW_DATASET_FILENAME)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop(columns=[c for c in ["Unnamed: 0", "Unnamed: 0.1"] if c in df.columns])
    return df


def run(remove_hashtag_word: bool = True) -> pd.DataFrame:
    df = load_raw()

    n_before = len(df)
    df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)
    print(f"Duplicate posts removed: {n_before - len(df)} (kept {len(df)})")

    df["sentiment_3class"] = df["Sentiment"].apply(map_sentiment)
    unmapped = df.loc[df["sentiment_3class"] == "Unmapped", "Sentiment"].unique()
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} unmapped labels: {sorted(unmapped)}")

    results = df["Text"].apply(lambda t: preprocess_pipeline(t, remove_hashtag_word))
    for key in ["lexicon_text", "tfidf_tokens", "tfidf_text", "lstm_tokens", "lstm_text"]:
        df[key] = results.apply(lambda r: r[key])

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "sentimentdataset_preprocessed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nFinal class distribution:")
    print(df["sentiment_3class"].value_counts())

    return df


if __name__ == "__main__":
    run()