import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models" / "multipositive"

EMBEDDING_DIM = 32
BATCH_SIZE = 8192
EPOCHS = 3
TRAIN_SAMPLE_SIZE = None
RANDOM_SEED = 42



class RetrievalModel(tfrs.Model):
    def __init__(self, user_ids, movie_ids):
        super().__init__()

        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=user_ids,
                mask_token=None,
            ),
            tf.keras.layers.Embedding(
                len(user_ids) + 1,
                EMBEDDING_DIM,
            ),
        ])

        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=movie_ids,
                mask_token=None,
            ),
            tf.keras.layers.Embedding(
                len(movie_ids) + 1,
                EMBEDDING_DIM,
            ),
        ])

    def compute_loss(self, features, training=False):
        user_ids = features["userId"]

        user_embeddings = self.user_model(user_ids)
        movie_embeddings = self.movie_model(features["movieId"])

        # Score every user against every movie in the batch.
        scores = tf.matmul(
            user_embeddings,
            movie_embeddings,
            transpose_b=True,
        )

        # Every movie in the batch associated with the same user
        # is treated as a positive for that user's query.
        positive_mask = tf.equal(
            user_ids[:, tf.newaxis],
            user_ids[tf.newaxis, :],
        )

        # Log probability mass across all batch candidates.
        log_denominator = tf.reduce_logsumexp(scores, axis=1)

        # Log probability mass across all same-user positives.
        negative_infinity = tf.constant(-1e9, dtype=scores.dtype)
        positive_scores = tf.where(
            positive_mask,
            scores,
            negative_infinity,
        )
        log_numerator = tf.reduce_logsumexp(
            positive_scores,
            axis=1,
        )

        # Multi-positive contrastive loss.
        return tf.reduce_mean(log_denominator - log_numerator)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    print("Loading training data...")
    train = pd.read_parquet(
        train_path,
        columns=["userId", "movieId", "rating"],
    )

    # Train retrieval on movies users rated positively.
    train = train[train["rating"] >= 4.0].copy()

    # The retrieval model only needs user and movie IDs.
    train = train[["userId", "movieId"]]

    if TRAIN_SAMPLE_SIZE is not None:
        sample_size = min(TRAIN_SAMPLE_SIZE, len(train))
        train = train.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        ).copy()
    else:
        train = train.copy()

    # String IDs support non-consecutive user and movie identifiers.
    train["userId"] = train["userId"].astype(str)
    train["movieId"] = train["movieId"].astype(str)

    user_ids = train["userId"].unique().tolist()
    movie_ids = train["movieId"].unique().tolist()

    print(f"Training interactions: {len(train):,}")
    print(f"Training users: {len(user_ids):,}")
    print(f"Training movies: {len(movie_ids):,}")
    print(f"Embedding dimensions: {EMBEDDING_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    dataset = tf.data.Dataset.from_tensor_slices({
        "userId": train["userId"].to_numpy(),
        "movieId": train["movieId"].to_numpy(),
    })

    
    dataset = (
        dataset
        .shuffle(
            buffer_size=100_000,
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Diagnostic: measure repeated users in the first batch.
    for batch in dataset.take(1):
        batch_users = batch["userId"].numpy().astype(str)
        unique_users = np.unique(batch_users)

        print("\n--- Batch Diagnostics ---")
        print(f"Interactions in batch: {len(batch_users):,}")
        print(f"Unique users in batch: {len(unique_users):,}")
        print(f"Repeated-user interactions: "
              f"{len(batch_users) - len(unique_users):,}")
        print(f"Unique-user percentage: "
              f"{len(unique_users) / len(batch_users) * 100:.2f}%")


    model = RetrievalModel(user_ids, movie_ids)
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(0.05)
    )

    print("\nStarting retrieval model training...")

    history = model.fit(
        dataset,
        epochs=EPOCHS,
        verbose=1,
    )

    print("\n--- Training Loss Summary ---")
    for epoch, loss in enumerate(history.history["loss"], start=1):
        print(f"Epoch {epoch}: loss = {loss:.4f}")

    # Save the embedding models and their matching ID vocabularies.
    model.user_model.save(MODEL_DIR / "user_embedding.keras")
    model.movie_model.save(MODEL_DIR / "movie_embedding.keras")

    np.save(
        MODEL_DIR / "user_ids.npy",
        np.asarray(user_ids, dtype=str),
    )
    np.save(
        MODEL_DIR / "movie_ids.npy",
        np.asarray(movie_ids, dtype=str),
    )

    print("\nRetrieval training completed.")
    print(f"Saved user embeddings model: {MODEL_DIR / 'user_embedding.keras'}")
    print(f"Saved movie embeddings model: {MODEL_DIR / 'movie_embedding.keras'}")
    print(f"Saved ID vocabularies and artifacts in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
