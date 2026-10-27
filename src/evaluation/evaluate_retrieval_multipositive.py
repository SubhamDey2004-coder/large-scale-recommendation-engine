
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models" / "multipositive"

K = 10
MAX_USERS = 500
RANDOM_SEED = 42
RELEVANT_RATING = 4.0
CANDIDATE_COUNT = 500


def recall_at_k(recommended, relevant):
    """Recall@K: fraction of relevant items found in recommendations."""
    if not relevant:
        return 0.0

    return len(set(recommended[:K]) & relevant) / len(relevant)


def main():
    # Load trained retrieval model and indexed movie catalogue.
    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )

    user_ids = set(
        np.load(
            MODEL_DIR / "user_ids.npy",
            allow_pickle=False,
        ).tolist()
    )

    movie_ids = np.load(
        MODEL_DIR / "movie_ids.npy",
        allow_pickle=False,
    ).astype(np.int64)

    indexed_movies = set(movie_ids.tolist())
    index = faiss.read_index(str(MODEL_DIR / "movies_multipositive.faiss"))
    if index.ntotal != len(movie_ids):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but movie_ids.npy contains {len(movie_ids)} IDs."
        )

    # Load chronological train/test splits.
    train = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["userId", "movieId"],
    )

    test = pd.read_parquet(
        DATA_DIR / "validation.parquet",
        columns=["userId", "movieId", "rating"],
    )

    train["userId"] = train["userId"].astype(str)
    test["userId"] = test["userId"].astype(str)

    train["movieId"] = train["movieId"].astype(np.int64)
    test["movieId"] = test["movieId"].astype(np.int64)

    # Training history is used to exclude previously seen movies.
    train_history = train.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )

    # Popularity baseline is computed using training interactions only.
    train_known = train[train["movieId"].isin(indexed_movies)]

    popular_movies = (
        train_known["movieId"]
        .value_counts()
        .index.astype(int)
        .tolist()
    )

    # Measure catalogue coverage BEFORE filtering test interactions.
    relevant_test = test[test["rating"] >= RELEVANT_RATING]

    relevant_test_known_users = relevant_test[
        relevant_test["userId"].isin(user_ids)
    ]

    relevant_test_indexed = relevant_test_known_users[
        relevant_test_known_users["movieId"].isin(indexed_movies)
    ]

    missing_movie_rows = relevant_test_known_users[
        ~relevant_test_known_users["movieId"].isin(indexed_movies)
    ]

    print("\n--- Evaluation Diagnostics ---")
    print(f"Trained users: {len(user_ids):,}")
    print(f"Indexed movies: {len(indexed_movies):,}")
    print(f"FAISS vectors: {index.ntotal:,}")
    print(f"All relevant test interactions: {len(relevant_test):,}")
    print(
        "Relevant interactions from trained users:",
        f"{len(relevant_test_known_users):,}",
    )
    print(
        "Relevant interactions covered by index:",
        f"{len(relevant_test_indexed):,}",
    )
    print(
        "Relevant test interactions missing from index:",
        f"{len(missing_movie_rows):,}",
    )
    print(
        "Unique relevant test movies missing from index:",
        f"{missing_movie_rows['movieId'].nunique():,}",
    )

    if relevant_test_indexed.empty:
        print("No eligible relevant test interactions found.")
        return

    # Build each user's relevant held-out movie set.
    user_groups = relevant_test_indexed.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )

    eligible_users = user_groups[user_groups.map(len) > 0]

    rng = np.random.default_rng(RANDOM_SEED)

    selected_users = rng.choice(
        eligible_users.index.to_numpy(),
        size=min(MAX_USERS, len(eligible_users)),
        replace=False,
    )

    retrieval_recalls = []
    popularity_recalls = []
    relevant_counts = []
    retrieval_counts = []
    popularity_counts = []

    for user_id in selected_users:
        relevant = eligible_users[user_id]
        seen = train_history.get(user_id, set())

        # Retrieve candidate movies for this user's embedding.
        embedding = user_model(
            tf.constant([user_id]),
            training=False,
        ).numpy().astype(np.float32)

        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                f"Unexpected user embedding shape: {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                f"Non-finite user embedding for user {user_id}"
            )

        faiss.normalize_L2(embedding)

        candidate_count = min(CANDIDATE_COUNT, index.ntotal)
        _, positions = index.search(embedding, candidate_count)

        recommended = []

        for pos in positions[0]:
            if pos < 0:
                continue

            # The FAISS position maps to the same position in movie_ids.
            movie_id = int(movie_ids[pos])

            if movie_id not in seen:
                recommended.append(movie_id)

            if len(recommended) == K:
                break

        # Popularity baseline with the same seen-item exclusion.
        popular_for_user = [
            movie_id
            for movie_id in popular_movies
            if movie_id not in seen
        ][:K]

        relevant_counts.append(len(relevant))
        retrieval_counts.append(len(recommended))
        popularity_counts.append(len(popular_for_user))

        retrieval_recalls.append(
            recall_at_k(recommended, relevant)
        )

        popularity_recalls.append(
            recall_at_k(popular_for_user, relevant)
        )

    # Report results and candidate-coverage diagnostics.
    print("\n--- Retrieval Evaluation Results ---")
    print(f"Evaluated users: {len(selected_users)}")
    print(f"Relevance threshold: rating >= {RELEVANT_RATING}")
    print(
        "Average relevant test movies per user:",
        f"{np.mean(relevant_counts):.2f}",
    )
    print(
        "Average unseen recommendations returned:",
        f"{np.mean(retrieval_counts):.2f}",
    )
    print(
        "Users receiving fewer than 10 personalized recommendations:",
        sum(count < K for count in retrieval_counts),
    )
    print(
        "Average popularity recommendations returned:",
        f"{np.mean(popularity_counts):.2f}",
    )
    print(
        f"Personalized Recall@{K}:",
        f"{np.mean(retrieval_recalls):.4f}",
    )
    print(
        f"Popularity Recall@{K}:",
        f"{np.mean(popularity_recalls):.4f}",
    )

    print(
        "\nEvaluation: chronological test split, "
        "known users, indexed relevant movies, "
        "and training-history exclusion."
    )
    print(
        "Note: preliminary offline evaluation; "
        "results are not evidence of production quality."
    )


if __name__ == "__main__":
    main()
