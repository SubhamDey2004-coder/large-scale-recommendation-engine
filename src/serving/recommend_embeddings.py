
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"


@lru_cache(maxsize=1)
def load_resources():
    """Load retrieval and embedding-ranking resources once."""
    print("Loading embedding-based recommendation resources...")

    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    movie_model = tf.keras.models.load_model(
        MODEL_DIR / "movie_embedding.keras"
    )
    ranker = keras.models.load_model(
        MODEL_DIR / "ranking_model_embeddings.keras"
    )

    user_ids = np.load(
        MODEL_DIR / "user_ids.npy", allow_pickle=False
    ).astype(str)
    movie_ids = np.load(
        MODEL_DIR / "movie_ids.npy", allow_pickle=False
    ).astype(np.int64)

    index = faiss.read_index(str(MODEL_DIR / "movies.faiss"))

    if index.ntotal != len(movie_ids):
        raise ValueError("FAISS index and movie ID list do not match.")

    movie_features = pd.read_parquet(
        PROCESSED_DIR / "movie_features.parquet"
    )
    user_history = pd.read_parquet(
        PROCESSED_DIR / "user_history.parquet"
    )

    genre_columns = [
        col for col in movie_features.columns
        if col.startswith("genre_")
    ]

    if len(genre_columns) != 19:
        raise ValueError(
            f"Expected 19 genre columns, found {len(genre_columns)}."
        )

    if ranker.input_shape[-1] != 83:
        raise ValueError(
            f"Expected 83 ranker inputs, got {ranker.input_shape[-1]}."
        )

    return {
        "user_model": user_model,
        "movie_model": movie_model,
        "ranker": ranker,
        "user_ids": set(user_ids.tolist()),
        "movie_ids": movie_ids,
        "index": index,
        "movie_features": movie_features,
        "user_history": user_history,
        "genre_columns": genre_columns,
    }


def recommend(user_id, top_k=10, candidate_count=200):
    """Retrieve candidates, remove seen movies, then rerank them."""
    if top_k < 1 or candidate_count < 1:
        raise ValueError("top_k and candidate_count must be positive.")

    r = load_resources()
    user_id_str = str(user_id)

    if user_id_str not in r["user_ids"]:
        raise ValueError(
            f"User {user_id} is not present in the trained retrieval model."
        )

    user_vector = r["user_model"](
        tf.constant([user_id_str]),
        training=False,
    ).numpy().astype(np.float32)

    retrieval_vector = user_vector.copy()
    faiss.normalize_L2(retrieval_vector)

    search_count = min(candidate_count, r["index"].ntotal)
    _, positions = r["index"].search(retrieval_vector, search_count)

    candidates = [
        int(r["movie_ids"][pos])
        for pos in positions[0]
        if 0 <= pos < len(r["movie_ids"])
    ]

    history = r["user_history"]
    user_row = history.loc[
        history["userId"] == int(user_id), "seen_movies"
    ]
    seen = set(user_row.iloc[0]) if not user_row.empty else set()

    candidates = list(dict.fromkeys(
        movie_id for movie_id in candidates if movie_id not in seen
    ))

    if not candidates:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    genre_columns = r["genre_columns"]
    candidate_df = pd.DataFrame({"movieId": candidates})

    candidate_df = candidate_df.merge(
        r["movie_features"][["movieId", "title"] + genre_columns],
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    candidate_df[genre_columns] = candidate_df[genre_columns].fillna(0)

    movie_vectors = r["movie_model"](
        tf.constant(candidate_df["movieId"].astype(str).to_numpy()),
        training=False,
    ).numpy().astype(np.float32)

    user_vectors = np.repeat(user_vector, len(candidate_df), axis=0)
    genre_values = candidate_df[genre_columns].to_numpy(dtype=np.float32)

    # Feature order used during training: user embeddings, movie
    # embeddings, then genre features.
    X = np.column_stack([
        user_vectors,
        movie_vectors,
        genre_values,
    ]).astype(np.float32)

    if X.shape[1] != r["ranker"].input_shape[-1]:
        raise ValueError(f"Unexpected ranker feature count: {X.shape[1]}")

    candidate_df["score"] = r["ranker"].predict(
        X, batch_size=4096, verbose=0
    ).ravel()

    return (
        candidate_df.sort_values("score", ascending=False)
        [["movieId", "title", "score"]]
        .head(top_k)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    resources = load_resources()
    example_user = next(iter(sorted(resources["user_ids"])))

    results = recommend(example_user, top_k=10, candidate_count=200)
    print(f"\nRecommendations for user {example_user}:\n")
    print(results.to_string(index=False))
