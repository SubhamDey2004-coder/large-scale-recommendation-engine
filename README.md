# Large-Scale Personalized Recommendation Engine

A two-stage movie recommendation system built with **TensorFlow, FAISS, and FastAPI** using the MovieLens 25M dataset.

The system retrieves candidate movies using learned user and movie embeddings, then reranks them using a neural ranking model to generate personalized recommendations.

## Features

- **Large-scale dataset:** Processes MovieLens 25M rating data.
- **User and movie embeddings:** Learns 32-dimensional representations.
- **Candidate retrieval:** Uses FAISS for efficient nearest-neighbor search.
- **Personalized ranking:** Reranks candidates using user embeddings, movie embeddings, and genre features.
- **Seen-item filtering:** Excludes movies already present in a user's viewing history.
- **REST API:** Exposes recommendations through FastAPI.
- **Evaluation:** Compares retrieval-only recommendations with embedding-based reranking.

## Architecture

```text
MovieLens 25M Dataset
        |
        v
Data Preparation and Time-Based Split
        |
        v
User and Movie Embedding Training
        |
        v
FAISS Candidate Retrieval
        |
        v
Seen-Movie Filtering
        |
        v
Embedding-Based Neural Ranker
        |
        v
Top-K Personalized Recommendations
        |
        v
FastAPI REST API
```

## Evaluation Results

On the project's fixed validation sample of 500 users:

| Metric | Result |
|---|---:|
| Retrieval-only Recall@10 | 0.0169 |
| Retrieval + embedding ranker Recall@10 | 0.0431 |
| Absolute improvement | +0.0262 |
| Relative improvement | Approximately 155% |

The embedding-based ranker improved Recall@10 by approximately **2.55×** over retrieval-only recommendations in this evaluation.

These results are from a fixed validation sample. They should not be interpreted as a guarantee of performance on all users or on the held-out test set.

The ranker also achieved a classification ROC-AUC of **0.7884** and accuracy of **0.7146** on its separate classification validation split. These classification metrics are distinct from recommendation Recall@10.

## Technology Stack

- Python 3.11
- TensorFlow / Keras
- FAISS
- Pandas and NumPy
- FastAPI and Uvicorn
- PyArrow / Parquet

## Project Structure

```text
src/
├── data/          # Dataset download, preparation, and user history
├── features/      # Movie feature construction
├── retrieval/     # Embedding training and FAISS indexing
├── ranking/       # Ranking feature preparation and model training
├── evaluation/    # Retrieval and ranking evaluation
└── serving/       # Recommendation pipeline and REST API

artifacts/models/  # Trained models, embeddings, and FAISS index
data/processed/    # Processed datasets and features
configs/           # Project configuration
docs/              # Project documentation
tests/             # Tests
```

## Setup

Use Python 3.11 and create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The trained models, FAISS index, and processed data must be available in the paths expected by the serving code. The original MovieLens dataset is large, so ensure adequate disk space before downloading or rebuilding the pipeline.

## Run the API

Start the API from the project root:

```powershell
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation:

- `http://127.0.0.1:8000/docs`

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/recommend/{user_id}?top_k=10` | Generate personalized recommendations |

Example request:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/1?top_k=10"
```

The `top_k` parameter accepts values from 1 to 50. Unknown users return HTTP 404, while invalid parameter values are rejected by FastAPI.

## Current Status

The main retrieval, ranking, evaluation, and serving components have been implemented. The embedding-based recommendation pipeline has been tested locally through both direct Python calls and the FastAPI endpoint.

Remaining work includes broader evaluation on the held-out test set, automated test coverage, reproducibility checks, and final repository cleanup.

## Limitations

- New users and users absent from the trained vocabulary are not currently supported by the embedding-based recommendation endpoint.
- Recommendation quality depends on the available user history, indexed movies, and learned embeddings.
- Validation metrics are based on a fixed sample and require broader testing before making general performance claims.
- Redis caching, scheduled retraining, and cold-start fallbacks are planned extensions, not currently implemented features.

## License and Dataset

Review the MovieLens dataset terms and the project's chosen software license before distributing the dataset or publishing the repository.
