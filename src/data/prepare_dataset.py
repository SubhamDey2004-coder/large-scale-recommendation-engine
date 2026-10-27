
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(RAW_DIR / "ratings.csv")

    # Sort chronologically to avoid future-data leakage.
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    n = len(ratings)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train = ratings.iloc[:train_end]
    validation = ratings.iloc[train_end:validation_end]
    test = ratings.iloc[validation_end:]

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    validation.to_parquet(
        PROCESSED_DIR / "validation.parquet", index=False
    )
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print(f"Total ratings: {n:,}")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        start = pd.to_datetime(split["timestamp"].min(), unit="s")
        end = pd.to_datetime(split["timestamp"].max(), unit="s")
        print(f"{name} dates: {start} to {end}")

    print(f"\nSaved splits to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
