
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
OUTPUT_PATH = ROOT / "data" / "processed" / "user_history.parquet"

ratings_path = RAW_DIR / "ratings.csv"

print("Reading user and movie IDs from MovieLens 25M...")

ratings = pd.read_csv(
    ratings_path,
    usecols=["userId", "movieId"],
    dtype={"userId": "int32", "movieId": "int32"},
)

history = ratings.groupby("userId")["movieId"].apply(list)

user_history = pd.DataFrame({
    "userId": history.index.astype("int32"),
    "seen_movies": history.values,
})

user_history.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved history for {len(user_history):,} users.")
print(f"Output: {OUTPUT_PATH}")
