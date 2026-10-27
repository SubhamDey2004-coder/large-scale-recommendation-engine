
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import faiss
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "models"


def main():
    movie_model = tf.keras.models.load_model(
        MODEL_DIR / "movie_embedding.keras"
    )
    movie_ids = np.load(MODEL_DIR / "movie_ids.npy", allow_pickle=False)

    # Generate an embedding for every movie in the trained vocabulary.
    movie_ids_tensor = tf.constant(movie_ids.astype(str))
    embeddings = movie_model(movie_ids_tensor, training=False).numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Cosine similarity: normalize vectors, then use inner product.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    print("Movie ID count:", len(movie_ids))
    print("Indexed vector count:", index.ntotal)
    print("Embedding shape:", embeddings.shape)
    print("All embeddings finite:", np.isfinite(embeddings).all())
    print(
        "Embedding norm range:",
        np.linalg.norm(embeddings, axis=1).min(),
        "to",
        np.linalg.norm(embeddings, axis=1).max(),
    )

    index_path = MODEL_DIR / "movies.faiss"
    faiss.write_index(index, str(index_path))

    print(f"Movies indexed: {index.ntotal:,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"FAISS index saved: {index_path}")


if __name__ == "__main__":
    main()
