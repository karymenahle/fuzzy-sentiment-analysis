# Fuzzy Sentiment Analysis

MSc Dissertation project (6G7V0007) — Manchester Metropolitan University, MSc Artificial Intelligence.

**Author:** Karyme Nahle Acosta
**Supervisor:** Naomi Adel

## Project Title

Modelling Linguistic Variables and Fuzzy Rules for Social Media Sentiment Analysis

## Research Question

Under controlled conditions on the same dataset and evaluation protocol, does a fuzzy logic-based sentiment analysis system — incorporating linguistic variables and fuzzy IF-THEN rules — achieve competitive or superior classification performance compared to SVM, Naive Bayes, and LSTM baselines, particularly on ambiguous and borderline cases?

## Overview

This project implements and compares four sentiment classification approaches on the same multi-platform social media dataset (Twitter, Instagram, Facebook):

1. **Fuzzy Logic System** — linguistic variables (positive intensity, negative intensity, subjectivity) and fuzzy IF-THEN rules, implemented with `scikit-fuzzy`, using VADER polarity scores as inputs.
2. **Naive Bayes** — TF-IDF features with Multinomial Naive Bayes (`scikit-learn`).
3. **SVM** — TF-IDF features with a linear-kernel SVM (`scikit-learn`).
4. **LSTM** — sequential deep learning model with word embeddings (TensorFlow/Keras).

All four systems are evaluated using stratified k-fold cross-validation on precision, recall, F1-score, and accuracy, with dedicated analysis of performance on ambiguous/borderline cases.

## Dataset

[Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) (Parmar, 2024) — Kaggle, public licence.

Raw data is **not** committed to this repository (see `.gitignore`). Download the dataset and place it in `data/raw/` before running preprocessing.

## Repository Structure

```
fuzzy-sentiment-analysis/
├── data/
│   ├── raw/              # Original dataset (not version-controlled)
│   └── processed/        # Cleaned/preprocessed data (not version-controlled)
├── notebooks/            # Exploratory analysis and prototyping
├── src/
│   ├── preprocessing/    # Tokenisation, cleaning, TF-IDF, embeddings
│   ├── fuzzy_system/     # Linguistic variables, fuzzy rules, inference
│   ├── baselines/        # Naive Bayes, SVM, LSTM implementations
│   ├── evaluation/       # Cross-validation, metrics, borderline-case analysis
│   └── utils/            # Shared helpers (config, logging, I/O)
├── results/
│   ├── figures/          # Generated plots
│   └── metrics/          # Saved evaluation results
├── tests/                # Unit tests
├── docs/                 # ToR, ethics approval, dissertation drafts/notes
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/<your-username>/fuzzy-sentiment-analysis.git
cd fuzzy-sentiment-analysis
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download required NLTK/spaCy resources (run once):

```bash
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
```

## Usage

```bash
# 1. Place the raw Kaggle CSV in data/raw/
# 2. Run preprocessing
python -m src.preprocessing.run

# 3. Train baselines
python -m src.baselines.run_all

# 4. Run the fuzzy inference system
python -m src.fuzzy_system.run

# 5. Evaluate and compare all four systems
python -m src.evaluation.run
```

## Project Status

This project is under active development as part of an MSc dissertation (project window: 01/06/2026 – 11/09/2026).

## Ethics

This project does not involve human participants. All data used is publicly available and anonymised. EthOS approval reference: TBC.

## References

See `docs/` for the full Terms of Reference and reference list, including Zadeh (1975), Parmar (2024), and related fuzzy-NLP literature.

## Licence

TBC — for academic/dissertation purposes.
