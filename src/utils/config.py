"""Project-wide configuration: paths and constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"

RAW_DATASET_FILENAME = "sentimentdataset.csv"

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

RANDOM_SEED = 42
N_FOLDS = 5
