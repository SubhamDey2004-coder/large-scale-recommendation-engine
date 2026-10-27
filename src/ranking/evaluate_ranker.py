
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"


def main():
    df = pd.read_parquet(DATA_DIR / "ranking_train.parquet")

    genre_columns = np.load(
        MODEL_DIR / "ranking_genre_columns.npy",
        allow_pickle=True,
    ).tolist()

    user_scaler = joblib.load(MODEL_DIR / "user_scaler.joblib")
    movie_scaler = joblib.load(MODEL_DIR / "movie_scaler.joblib")
    model = tf.keras.models.load_model(
        MODEL_DIR / "ranking_model.keras"
    )

    # Reproduce the same held-out split used during training.
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    _, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=df["label"].to_numpy(),
    )

    val = df.iloc[val_indices]

    user_features = user_scaler.transform(
        val["userId"].to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    movie_features = movie_scaler.transform(
        val["movieId"].to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    genres = val[genre_columns].to_numpy(dtype=np.float32)

    X_val = np.column_stack([
        user_features,
        movie_features,
        genres,
    ]).astype(np.float32)

    y_true = val["label"].to_numpy(dtype=np.int32)
    probabilities = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(np.int32)

    print(f"Validation examples: {len(y_true):,}")
    print(f"ROC-AUC:  {roc_auc_score(y_true, probabilities):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision:{precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"Recall:   {recall_score(y_true, predictions, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
