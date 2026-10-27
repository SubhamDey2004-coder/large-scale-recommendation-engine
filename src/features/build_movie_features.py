
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = ROOT / "data" / "processed"

MOVIES_PATH = RAW_DIR / "movies.csv"
OUTPUT_PATH = PROCESSED_DIR / "movie_features.parquet"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(MOVIES_PATH)

    movies["genres_list"] = (
        movies["genres"]
        .fillna("")
        .apply(
            lambda value: [
                genre
                for genre in value.split("|")
                if genre and genre != "(no genres listed)"
            ]
        )
    )

    encoder = MultiLabelBinarizer()
    genre_features = encoder.fit_transform(movies["genres_list"])

    features = pd.DataFrame(
        genre_features,
        columns=[f"genre_{genre}" for genre in encoder.classes_],
    )

    result = pd.concat(
        [
            movies[["movieId", "title"]],
            features,
        ],
        axis=1,
    )

    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"Movies processed: {len(result):,}")
    print(f"Genre features: {len(encoder.classes_)}")
    print(f"Movies without listed genres: "
          f"{(result.iloc[:, 2:].sum(axis=1) == 0).sum():,}")
    print(f"Output saved: {OUTPUT_PATH}")
    print("\nSample features:")
    print(result.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
