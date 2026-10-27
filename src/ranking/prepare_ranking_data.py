
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

SAMPLE_SIZE = 500_000
RANDOM_SEED = 42


def main():
    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movie_features.parquet"

    train = pd.read_parquet(
        train_path,
        columns=["userId", "movieId", "rating", "timestamp"],
    )

    movies = pd.read_parquet(movies_path)
    genre_columns = [
        column for column in movies.columns
        if column.startswith("genre_")
    ]

    # Sample interactions for a first, manageable ranking experiment.
    sample = train.sample(
        n=min(SAMPLE_SIZE, len(train)),
        random_state=RANDOM_SEED,
    ).copy()

    # Binary target: ratings of 4 or above count as positive feedback.
    sample["label"] = (sample["rating"] >= 4.0).astype("int8")

    # Attach movie genres to each interaction.
    ranking_data = sample.merge(
        movies[["movieId"] + genre_columns],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    ranking_data[genre_columns] = (
        ranking_data[genre_columns].fillna(0).astype("int8")
    )

    output_path = DATA_DIR / "ranking_train.parquet"
    ranking_data.to_parquet(output_path, index=False)

    print(f"Training rows: {len(ranking_data):,}")
    print(f"Genre features: {len(genre_columns)}")
    print(f"Positive labels: {ranking_data['label'].sum():,}")
    print(f"Positive rate: {ranking_data['label'].mean():.3f}")
    print(f"Output saved: {output_path}")
    print("\nSample:")
    print(ranking_data.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
