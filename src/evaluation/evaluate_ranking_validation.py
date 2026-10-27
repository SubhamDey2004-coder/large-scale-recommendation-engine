
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import keras
import faiss
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"

K = 10
MAX_USERS = 500
SEED = 42
CANDIDATE_COUNT = 2000
EMBEDDING_DIM = 32


def recall_at_k(recommended, relevant):
    if not relevant:
        return 0.0
    return len(set(recommended[:K]) & relevant) / len(relevant)


def main():
    print("Loading models and FAISS index...")

    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    movie_model = tf.keras.models.load_model(
        MODEL_DIR / "movie_embedding.keras"
    )
    ranker = keras.models.load_model(
        MODEL_DIR / "ranking_model_embeddings.keras"
    )

    user_ids = set(
        np.load(
            MODEL_DIR / "user_ids.npy",
            allow_pickle=False,
        ).astype(str).tolist()
    )
    movie_ids = np.load(
        MODEL_DIR / "movie_ids.npy",
        allow_pickle=False,
    ).astype(np.int64)

    index = faiss.read_index(str(MODEL_DIR / "movies.faiss"))

    if index.ntotal != len(movie_ids):
        raise ValueError("FAISS index and movie ID list do not match.")

    if ranker.input_shape[-1] != 83:
        raise ValueError(
            f"Expected 83 ranker features, got {ranker.input_shape[-1]}."
        )

    print("Loading metadata and evaluation data...")

    movies = pd.read_parquet(DATA_DIR / "movie_features.parquet")
    genre_columns = [
        col for col in movies.columns if col.startswith("genre_")
    ]

    if len(genre_columns) != 19:
        raise ValueError(
            f"Expected 19 genre columns, found {len(genre_columns)}."
        )

    train = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["userId", "movieId"],
    )
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet",
        columns=["userId", "movieId", "rating"],
    )

    train["userId"] = train["userId"].astype(str)
    validation["userId"] = validation["userId"].astype(str)
    train["movieId"] = train["movieId"].astype(np.int64)
    validation["movieId"] = validation["movieId"].astype(np.int64)

    train_pairs = train.drop_duplicates(["userId", "movieId"])

    validation = validation.merge(
        train_pairs.assign(_in_train=True),
        on=["userId", "movieId"],
        how="left",
    )
    validation = validation[
        validation["_in_train"].isna()
    ].drop(columns="_in_train")

    seen_history = train.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )

    indexed_movies = set(movie_ids.tolist())

    relevant = validation[
        (validation["rating"] >= 4.0)
        & validation["userId"].isin(user_ids)
        & validation["movieId"].isin(indexed_movies)
    ]

    user_relevant = relevant.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )
    eligible = user_relevant[user_relevant.map(len) > 0]

    if eligible.empty:
        raise ValueError("No eligible validation users found.")

    rng = np.random.default_rng(SEED)
    selected = rng.choice(
        eligible.index.to_numpy(),
        size=min(MAX_USERS, len(eligible)),
        replace=False,
    )

    movie_metadata = (
        movies[["movieId"] + genre_columns]
        .drop_duplicates("movieId")
        .set_index("movieId")
    )

    retrieval_recalls = []
    reranked_recalls = []

    print(f"Evaluating {len(selected)} users...")

    for i, user_id in enumerate(selected, start=1):
        relevant_movies = eligible[user_id]
        seen = seen_history.get(user_id, set())

        user_vector = user_model(
            tf.constant([user_id]),
            training=False,
        ).numpy().astype(np.float32)

        # Retrieval uses the same normalized user vector as before.
        retrieval_vector = user_vector.copy()
        faiss.normalize_L2(retrieval_vector)

        _, positions = index.search(
            retrieval_vector,
            min(CANDIDATE_COUNT, index.ntotal),
        )

        candidates = []
        for position in positions[0]:
            if position < 0:
                continue

            movie_id = int(movie_ids[position])

            if movie_id not in seen:
                candidates.append(movie_id)

            if len(candidates) >= CANDIDATE_COUNT:
                break

        retrieval_top_k = candidates[:K]

        if not candidates:
            reranked_top_k = []
        else:
            candidate_df = pd.DataFrame({"movieId": candidates})

            candidate_df = candidate_df.merge(
                movie_metadata.reset_index(),
                on="movieId",
                how="left",
                validate="one_to_one",
            )

            candidate_df[genre_columns] = (
                candidate_df[genre_columns].fillna(0)
            )

            # Compute one movie embedding per candidate.
            movie_vectors = movie_model(
                tf.constant(
                    candidate_df["movieId"].astype(str).to_numpy()
                ),
                training=False,
            ).numpy().astype(np.float32)

            # Repeat this user's embedding for every candidate.
            user_vectors = np.repeat(
                user_vector,
                len(candidate_df),
                axis=0,
            )

            genre_values = candidate_df[
                genre_columns
            ].to_numpy(dtype=np.float32)

            # Feature order must match the ranker training script:
            # 32 user embeddings + 32 movie embeddings + 19 genres.
            X = np.column_stack(
                [user_vectors, movie_vectors, genre_values]
            ).astype(np.float32)

            if X.shape[1] != ranker.input_shape[-1]:
                raise ValueError(
                    f"Ranker expects {ranker.input_shape[-1]} features, "
                    f"but evaluator built {X.shape[1]}."
                )

            scores = ranker.predict(
                X,
                batch_size=4096,
                verbose=0,
            ).ravel()

            order = np.argsort(-scores)

            reranked_top_k = candidate_df.iloc[order][
                "movieId"
            ].astype(int).head(K).tolist()

        retrieval_recalls.append(
            recall_at_k(retrieval_top_k, relevant_movies)
        )
        reranked_recalls.append(
            recall_at_k(reranked_top_k, relevant_movies)
        )

        if i % 100 == 0:
            print(f"Processed {i}/{len(selected)} users")

    retrieval_recall = float(np.mean(retrieval_recalls))
    reranked_recall = float(np.mean(reranked_recalls))
    difference = reranked_recall - retrieval_recall

    print("\n--- Validation Results ---")
    print(f"Users evaluated: {len(selected)}")
    print(f"Candidate count: {CANDIDATE_COUNT}")
    print(f"Retrieval-only Recall@{K}: {retrieval_recall:.4f}")
    print(f"Retrieval + embedding ranker Recall@{K}: {reranked_recall:.4f}")
    print(f"Recall difference: {difference:+.4f}")

    if difference > 0:
        print("Result: The embedding ranker improved Recall@10.")
    elif difference < 0:
        print("Result: The embedding ranker reduced Recall@10.")
    else:
        print("Result: No change in Recall@10.")


if __name__ == "__main__":
    main()
