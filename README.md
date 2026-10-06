# Search Ranking Model

> **Status: Under active development — Phase 1 (Project Foundation) complete.**

This project implements an end-to-end **Learning-to-Rank** search system using
**LambdaMART** (via LightGBM) and evaluates it against a traditional **BM25**
retrieval baseline. Starting from raw query–document pairs, the pipeline covers
candidate retrieval, feature engineering, model training, ranking, and rigorous
offline evaluation using standard IR metrics (NDCG, MAP, MRR).

## Preliminary Architecture

```text
Query
  ↓
Candidate Retrieval        ← BM25 baseline
  ↓
Feature Engineering        ← query–document relevance signals
  ↓
LambdaMART                 ← pairwise / listwise LTR model
  ↓
Ranking                    ← re-ranked candidate list
  ↓
Evaluation                 ← NDCG@k, MAP, MRR
```

## Project Layout

```text
search-ranking-model/
├── data/
│   ├── raw/               # original, immutable datasets
│   └── processed/         # cleaned and featurised data
├── src/                   # core library code
├── scripts/               # one-off CLI scripts
├── tests/                 # pytest test suite
├── models/                # serialised model artefacts
├── results/               # evaluation outputs and plots
├── notebooks/             # exploratory Jupyter notebooks
├── app/                   # Streamlit demo application
├── docs/                  # project documentation
├── config.yaml            # centralised configuration
└── requirements.txt
```

## Phases (planned)

| Phase | Description |
|-------|-------------|
| 1 | Project foundation ✅ |
| 2 | Data ingestion & EDA |
| 3 | Feature engineering |
| 4 | BM25 baseline |
| 5 | LambdaMART training |
| 6 | Evaluation & comparison |
| 7 | Streamlit demo |

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/sik04/Search-Ranking-Model.git
cd search-ranking-model

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## License

MIT
