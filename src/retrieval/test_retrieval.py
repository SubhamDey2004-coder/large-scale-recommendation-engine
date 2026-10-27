
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "ml-25m"
MODEL_DIR = ROOT / "artifacts" / "models"


def main():
    user_model = tf.keras.models.load_model(
        MODEL_DIR / "user_embedding.keras"
    )
    user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=False)
    movie_ids = np.load(MODEL_DIR / "movie_ids.npy", allow_pickle=False)
    index = faiss.read_index(str(MODEL_DIR / "movies.faiss"))

    # Select a user represented in the trained model.
    user_id = str(user_ids[0])
    user_embedding = user_model(
        tf.constant([user_id]), training=False
    ).numpy().astype(np.float32)

    faiss.normalize_L2(user_embedding)
    scores, positions = index.search(user_embedding, 10)

    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movie_lookup = movies.set_index("movieId")

    print(f"Recommendations for user {user_id}:\n")

    for score, position in zip(scores[0], positions[0]):
        movie_id = int(movie_ids[position])
        title = movie_lookup.loc[movie_id, "title"]
        print(f"{title} | similarity: {score:.4f}")


if __name__ == "__main__":
    main()
