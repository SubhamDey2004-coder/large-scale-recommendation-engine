
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"

SAMPLE_USERS = 500
NEGATIVES_PER_USER = 100
RANDOM_SEED = 42


def main():
    # Load trained embedding models and their ID vocabularies.
    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    movie_model = tf.keras.models.load_model(
        MODEL_DIR / "movie_embedding.keras"
    )

    user_ids = np.load(
        MODEL_DIR / "user_ids.npy",
        allow_pickle=False,
    ).astype(str)

    movie_ids = np.load(
        MODEL_DIR / "movie_ids.npy",
        allow_pickle=False,
    ).astype(str)

    # Diagnose embedding health.
    print("--- Embedding Diagnostics ---")

    user_embeddings = user_model(
        tf.constant(user_ids[:min(10_000, len(user_ids))])
    ).numpy()

    movie_embeddings = movie_model(
        tf.constant(movie_ids[:min(10_000, len(movie_ids))])
    ).numpy()

    for name, embeddings in [
        ("Users", user_embeddings),
        ("Movies", movie_embeddings),
    ]:
        norms = np.linalg.norm(embeddings, axis=1)

        print(f"\n{name}")
        print("Shape:", embeddings.shape)
        print("All finite:", np.isfinite(embeddings).all())
        print(f"Mean norm: {norms.mean():.6f}")
        print(f"Norm standard deviation: {norms.std():.6f}")
        print(
            "Mean per-dimension standard deviation:",
            f"{embeddings.std(axis=0).mean():.6f}",
        )

    # Load held-out positive validation interactions.
    validation = pd.read_parquet(
        DATA_DIR / "validation.parquet",
        columns=["userId", "movieId", "rating"],
    )

    validation = validation[
        validation["rating"] >= 4.0
    ].copy()

    validation["userId"] = validation["userId"].astype(str)
    validation["movieId"] = validation["movieId"].astype(str)

    user_set = set(user_ids.tolist())
    movie_set = set(movie_ids.tolist())

    validation = validation[
        validation["userId"].isin(user_set)
        & validation["movieId"].isin(movie_set)
    ]

    eligible_users = validation["userId"].unique()

    rng = np.random.default_rng(RANDOM_SEED)

    selected_users = rng.choice(
        eligible_users,
        size=min(SAMPLE_USERS, len(eligible_users)),
        replace=False,
    )

    movie_lookup = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    relevant_by_user = validation.groupby("userId")["movieId"].apply(
        lambda values: set(values)
    )

    # Compute embeddings for the complete indexed movie catalogue.
    print("\nComputing movie embeddings...")

    all_movie_embeddings = movie_model(
        tf.constant(movie_ids)
    ).numpy().astype(np.float32)

    positive_scores = []
    random_scores = []
    auc_scores = []

    rng = np.random.default_rng(RANDOM_SEED)

    # Evaluate sampled users against the full movie catalogue.
    for user_id in selected_users:
        user_embedding = user_model(
            tf.constant([user_id])
        ).numpy()[0]

        relevant_movies = relevant_by_user[user_id]

        relevant_indices = np.array(
            [
                movie_lookup[movie_id]
                for movie_id in relevant_movies
                if movie_id in movie_lookup
            ],
            dtype=np.int64,
        )

        if len(relevant_indices) == 0:
            continue

        # Score every indexed movie for this user.
        scores = all_movie_embeddings @ user_embedding

        user_positive_scores = scores[relevant_indices]
        positive_scores.extend(user_positive_scores.tolist())

        # Sample catalogue movies that are not validation positives.
        negative_pool = np.setdiff1d(
            np.arange(len(movie_ids)),
            relevant_indices,
            assume_unique=False,
        )

        if len(negative_pool) == 0:
            continue

        negative_indices = rng.choice(
            negative_pool,
            size=min(NEGATIVES_PER_USER, len(negative_pool)),
            replace=False,
        )

        user_negative_scores = scores[negative_indices]
        random_scores.extend(user_negative_scores.tolist())

        # Calculate sampled per-user ROC-AUC.
        y_true = np.concatenate(
            [
                np.ones(len(user_positive_scores)),
                np.zeros(len(user_negative_scores)),
            ]
        )

        y_scores = np.concatenate(
            [
                user_positive_scores,
                user_negative_scores,
            ]
        )

        auc_scores.append(
            roc_auc_score(y_true, y_scores)
        )

    # Report ranking quality.
    print("\n--- Ranking Quality Diagnostic ---")

    if auc_scores:
        print(f"Users evaluated for AUC: {len(auc_scores)}")
        print(f"Mean per-user AUC: {np.mean(auc_scores):.4f}")
        print(f"Median per-user AUC: {np.median(auc_scores):.4f}")
    else:
        print("AUC could not be calculated.")

    # Report positive-versus-random score distributions.
    print("\n--- Validation Score Diagnostics ---")
    print(f"Users sampled: {len(selected_users)}")

    if positive_scores and random_scores:
        print(f"Positive pairs scored: {len(positive_scores):,}")
        print(f"Random pairs scored: {len(random_scores):,}")

        print(
            "Mean positive score:",
            f"{np.mean(positive_scores):.6f}",
        )
        print(
            "Mean random score:",
            f"{np.mean(random_scores):.6f}",
        )
        print(
            "Positive score median:",
            f"{np.median(positive_scores):.6f}",
        )
        print(
            "Random score median:",
            f"{np.median(random_scores):.6f}",
        )
        print(
            "Positive score greater than random-pair mean:",
            f"{np.mean(positive_scores) > np.mean(random_scores)}",
        )
    else:
        print("Insufficient positive or random pairs for comparison.")


if __name__ == "__main__":
    main()
