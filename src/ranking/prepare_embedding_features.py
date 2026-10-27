import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"

OUTPUT_PATH = DATA_DIR / "ranking_embedding_features.parquet"
BATCH_SIZE = 8192


def main():
    print("Loading embedding models...")

    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    movie_model = tf.keras.models.load_model(
        MODEL_DIR / "movie_embedding.keras"
    )

    user_ids = np.load(
        MODEL_DIR / "user_ids.npy", allow_pickle=False
    ).astype(str)
    movie_ids = np.load(
        MODEL_DIR / "movie_ids.npy", allow_pickle=False
    ).astype(np.int64)

    print("Loading ranking interactions...")
    df = pd.read_parquet(
        DATA_DIR / "ranking_train.parquet"
    )

    # Keep only IDs represented by the trained embedding models.
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(np.int64)

    valid_users = set(user_ids.tolist())
    valid_movies = set(movie_ids.tolist())

    df = df[
        df["userId"].isin(valid_users)
        & df["movieId"].isin(valid_movies)
    ].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No interactions have valid user and movie embeddings.")

    print(f"Rows with valid embeddings: {len(df):,}")

    # Build ID-to-row maps to avoid repeated model inference per interaction.
    user_lookup = {value: i for i, value in enumerate(user_ids)}
    movie_lookup = {int(value): i for i, value in enumerate(movie_ids)}

    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    print("Computing user embeddings...")
    user_vectors = user_model(
        tf.constant(unique_users),
        training=False,
    ).numpy().astype(np.float32)

    print("Computing movie embeddings...")
    movie_vectors = movie_model(
        tf.constant(unique_movies.astype(str)),
        training=False,
    ).numpy().astype(np.float32)

    user_vector_lookup = {
        user_id: user_vectors[i]
        for i, user_id in enumerate(unique_users)
    }
    movie_vector_lookup = {
        int(movie_id): movie_vectors[i]
        for i, movie_id in enumerate(unique_movies)
    }

    user_matrix = np.stack(
        [user_vector_lookup[user_id] for user_id in df["userId"]]
    ).astype(np.float32)

    movie_matrix = np.stack(
        [movie_vector_lookup[int(movie_id)] for movie_id in df["movieId"]]
    ).astype(np.float32)

    user_columns = [f"user_emb_{i}" for i in range(user_matrix.shape[1])]
    movie_columns = [f"movie_emb_{i}" for i in range(movie_matrix.shape[1])]

    user_df = pd.DataFrame(user_matrix, columns=user_columns)
    movie_df = pd.DataFrame(movie_matrix, columns=movie_columns)

    genre_columns = [
        col for col in df.columns if col.startswith("genre_")
    ]

    output = pd.concat(
        [
            df[["userId", "movieId", "label"]].reset_index(drop=True),
            user_df,
            movie_df,
            df[genre_columns].reset_index(drop=True).astype(np.float32),
        ],
        axis=1,
    )

    output.to_parquet(OUTPUT_PATH, index=False)

    print(f"Output rows: {len(output):,}")
    print(f"User embedding features: {len(user_columns)}")
    print(f"Movie embedding features: {len(movie_columns)}")
    print(f"Genre features: {len(genre_columns)}")
    print(f"Total features excluding IDs and label: {len(user_columns) + len(movie_columns) + len(genre_columns)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()