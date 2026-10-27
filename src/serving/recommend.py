
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from functools import lru_cache
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"


@lru_cache(maxsize=1)
def load_resources():
    """Load models and data once per process."""
    print("Loading recommendation resources...")

    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=False)
    movie_ids = np.load(MODEL_DIR / "movie_ids.npy", allow_pickle=False)
    index = faiss.read_index(str(MODEL_DIR / "movies.faiss"))

    movie_features = pd.read_parquet(
        PROCESSED_DIR / "movie_features.parquet"
    )
    user_history = pd.read_parquet(
        PROCESSED_DIR / "user_history.parquet"
    )
    genre_columns = np.load(
        MODEL_DIR / "ranking_genre_columns.npy",
        allow_pickle=True,
    ).tolist()

    user_scaler = joblib.load(MODEL_DIR / "user_scaler.joblib")
    movie_scaler = joblib.load(MODEL_DIR / "movie_scaler.joblib")
    ranker = tf_keras.models.load_model(
        MODEL_DIR / "ranking_model_legacy"
    )

    resources = {
        "user_model": user_model,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "index": index,
        "movie_features": movie_features,
        "user_history": user_history,
        "genre_columns": genre_columns,
        "user_scaler": user_scaler,
        "movie_scaler": movie_scaler,
        "ranker": ranker,
    }

    print("Recommendation resources loaded.")
    return resources


def recommend(user_id, top_k=10, candidate_count=200):
    r = load_resources()

    user_model = r["user_model"]
    user_ids = r["user_ids"]
    movie_ids = r["movie_ids"]
    index = r["index"]

    user_id_str = str(user_id)

    if user_id_str not in set(user_ids.astype(str)):
        raise ValueError(
            f"User {user_id} is not present in the trained retrieval model."
        )

    user_embedding = user_model(
        tf.constant([user_id_str]), training=False
    ).numpy().astype(np.float32)

    faiss.normalize_L2(user_embedding)
    candidate_count = min(candidate_count, index.ntotal)
    _, positions = index.search(user_embedding, candidate_count)

    candidates = [
        int(movie_ids[pos])
        for pos in positions[0]
        if 0 <= pos < len(movie_ids)
    ]

    user_history = r["user_history"]
    user_row = user_history.loc[
        user_history["userId"] == int(user_id), "seen_movies"
    ]
    seen = set(user_row.iloc[0]) if not user_row.empty else set()

    candidates = [movie_id for movie_id in candidates if movie_id not in seen]

    if not candidates:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    genre_columns = r["genre_columns"]
    candidate_df = pd.DataFrame({"movieId": candidates})
    candidate_df["userId"] = int(user_id)

    candidate_df = candidate_df.merge(
        r["movie_features"][["movieId", "title"] + genre_columns],
        on="movieId",
        how="left",
        validate="one_to_one",
    )
    candidate_df[genre_columns] = candidate_df[genre_columns].fillna(0)

    user_values = r["user_scaler"].transform(
        candidate_df["userId"].to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    movie_values = r["movie_scaler"].transform(
        candidate_df["movieId"].to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    genre_values = candidate_df[genre_columns].to_numpy(dtype=np.float32)

    X = np.column_stack([
        user_values,
        movie_values,
        genre_values,
    ]).astype(np.float32)

    candidate_df["score"] = r["ranker"].predict(X, verbose=0).ravel()

    return (
        candidate_df.sort_values("score", ascending=False)
        [["movieId", "title", "score"]]
        .head(top_k)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    resources = load_resources()
    example_user = int(resources["user_ids"][0])

    results = recommend(example_user, top_k=10)
    print(f"\nRecommendations for user {example_user}:\n")
    print(results.to_string(index=False))
