
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"

K = 10
MAX_USERS = 500
RANDOM_SEED = 42
RELEVANT_RATING = 4.0
CANDIDATE_COUNT = 2000


def recall_at_k(recommended, relevant):
    """Recall@K: fraction of relevant items found in top-K recommendations."""
    if not relevant:
        return 0.0

    return len(set(recommended[:K]) & relevant) / len(relevant)


def main():
    # ---------------------------------------------------------
    # 1. Load the original retrieval model and matching index
    # ---------------------------------------------------------
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
    index = faiss.read_index(str(MODEL_DIR / "movies.faiss"))

    if index.ntotal != len(movie_ids):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but movie_ids.npy contains {len(movie_ids)} IDs."
        )

    # ---------------------------------------------------------
    # 2. Load chronological training and validation splits
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. Check train/validation user-movie overlap
    # ---------------------------------------------------------
    train_pairs = pd.MultiIndex.from_frame(
        train[["userId", "movieId"]].drop_duplicates()
    )

    validation_pairs = pd.MultiIndex.from_frame(
        validation[["userId", "movieId"]].drop_duplicates()
    )

    overlap_count = len(train_pairs.intersection(validation_pairs))

    print("\n--- Split Integrity Check ---")
    print(f"Unique training user-movie pairs: {len(train_pairs):,}")
    print(
        "Unique validation user-movie pairs:",
        f"{len(validation_pairs):,}",
    )
    print(f"User-movie pairs shared across splits: {overlap_count:,}")

    if overlap_count:
        print(
            "Warning: overlapping pairs were found. "
            "They will be excluded from validation targets."
        )

        train_pair_frame = train[
            ["userId", "movieId"]
        ].drop_duplicates()

        validation = validation.merge(
            train_pair_frame.assign(_seen_in_train=True),
            on=["userId", "movieId"],
            how="left",
        )

        validation = validation[
            validation["_seen_in_train"].isna()
        ].drop(columns="_seen_in_train")

    # ---------------------------------------------------------
    # 4. Build training history for seen-item exclusion
    # ---------------------------------------------------------
    train_history = train.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )

    # Popularity baseline uses training interactions only.
    train_known = train[train["movieId"].isin(indexed_movies)]

    popular_movies = (
        train_known["movieId"]
        .value_counts()
        .index.astype(int)
        .tolist()
    )

    # ---------------------------------------------------------
    # 5. Determine relevant validation interactions
    # ---------------------------------------------------------
    relevant_validation = validation[
        validation["rating"] >= RELEVANT_RATING
    ]

    relevant_known_users = relevant_validation[
        relevant_validation["userId"].isin(user_ids)
    ]

    relevant_indexed = relevant_known_users[
        relevant_known_users["movieId"].isin(indexed_movies)
    ]

    missing_movie_rows = relevant_known_users[
        ~relevant_known_users["movieId"].isin(indexed_movies)
    ]

    print("\n--- Validation Diagnostics ---")
    print(f"Trained users: {len(user_ids):,}")
    print(f"Indexed movies: {len(indexed_movies):,}")
    print(f"FAISS vectors: {index.ntotal:,}")
    print(
        "All relevant validation interactions:",
        f"{len(relevant_validation):,}",
    )
    print(
        "Relevant interactions from trained users:",
        f"{len(relevant_known_users):,}",
    )
    print(
        "Relevant interactions covered by index:",
        f"{len(relevant_indexed):,}",
    )
    print(
        "Relevant validation interactions missing from index:",
        f"{len(missing_movie_rows):,}",
    )
    print(
        "Unique relevant validation movies missing from index:",
        f"{missing_movie_rows['movieId'].nunique():,}",
    )

    if relevant_indexed.empty:
        print("No eligible relevant validation interactions found.")
        return

    # One set of relevant held-out movies per user.
    user_groups = relevant_indexed.groupby("userId")["movieId"].apply(
        lambda values: set(values.tolist())
    )

    eligible_users = user_groups[user_groups.map(len) > 0]

    rng = np.random.default_rng(RANDOM_SEED)

    selected_users = rng.choice(
        eligible_users.index.to_numpy(),
        size=min(MAX_USERS, len(eligible_users)),
        replace=False,
    )

    # ---------------------------------------------------------
    # 6. Evaluate personalized retrieval and popularity
    # ---------------------------------------------------------
    retrieval_recalls = []
    popularity_recalls = []
    relevant_counts = []
    retrieval_counts = []
    popularity_counts = []

    for user_id in selected_users:
        relevant = eligible_users[user_id]
        seen = train_history.get(user_id, set())

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

        # Normalize query vector for cosine-style retrieval.
        faiss.normalize_L2(embedding)

        candidate_count = min(CANDIDATE_COUNT, index.ntotal)
        _, positions = index.search(embedding, candidate_count)

        recommended = []

        for pos in positions[0]:
            if pos < 0:
                continue

            # FAISS position corresponds to the same position in movie_ids.
            movie_id = int(movie_ids[pos])

            if movie_id not in seen:
                recommended.append(movie_id)

            if len(recommended) == K:
                break

        # Popularity baseline with identical seen-item exclusion.
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

    # ---------------------------------------------------------
    # 7. Report metrics
    # ---------------------------------------------------------
    print("\n--- Validation Retrieval Results ---")
    print(f"Evaluated users: {len(selected_users)}")
    print(f"Relevance threshold: rating >= {RELEVANT_RATING}")
    print(
        "Average relevant validation movies per user:",
        f"{np.mean(relevant_counts):.2f}",
    )
    print(
        "Average unseen personalized recommendations returned:",
        f"{np.mean(retrieval_counts):.2f}",
    )
    print(
        f"Users receiving fewer than {K} personalized recommendations:",
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
        "\nEvaluation: chronological validation split, "
        "known users, indexed relevant movies, "
        "and training-history exclusion."
    )
    print(
        "Note: preliminary offline evaluation; "
        "results are not evidence of production quality."
    )


if __name__ == "__main__":
    main()
