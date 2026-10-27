
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

BATCH_SIZE = 8192
SHUFFLE_BUFFER_SIZE = 100_000
RANDOM_SEED = 42


def main():
    train_path = DATA_DIR / "train.parquet"

    print("Loading positive training interactions...")

    train = pd.read_parquet(
        train_path,
        columns=["userId", "movieId", "rating"],
    )

    train = train.loc[
        train["rating"] >= 4.0,
        ["userId", "movieId"],
    ].copy()

    train["userId"] = train["userId"].astype(str)
    train["movieId"] = train["movieId"].astype(str)

    print(f"Positive interactions: {len(train):,}")
    print(f"Unique users: {train['userId'].nunique():,}")
    print(f"Unique movies: {train['movieId'].nunique():,}")

    # Reproduce the shuffle used by the training pipeline.
    shuffled = train.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Inspect the first batch without training anything.
    batch = shuffled.iloc[:BATCH_SIZE]

    user_counts = batch["userId"].value_counts()

    interactions = len(batch)
    unique_users = batch["userId"].nunique()
    repeated_interactions = interactions - unique_users

    print("\n--- First-Batch Diagnostics ---")
    print(f"Interactions: {interactions:,}")
    print(f"Unique users: {unique_users:,}")
    print(f"Repeated-user interactions: {repeated_interactions:,}")
    print(f"Unique-user percentage: {unique_users / interactions * 100:.2f}%")
    print(f"Users appearing more than once: {(user_counts > 1).sum():,}")
    print(f"Maximum interactions from one user: {user_counts.max()}")

    print("\n--- User Frequency in Batch ---")
    print("Users appearing once:", int((user_counts == 1).sum()))
    print("Users appearing 2-5 times:", int(user_counts.between(2, 5).sum()))
    print("Users appearing 6-10 times:", int(user_counts.between(6, 10).sum()))
    print("Users appearing more than 10 times:", int((user_counts > 10).sum()))


if __name__ == "__main__":
    main()
