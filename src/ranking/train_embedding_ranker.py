
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "ranking_embedding_features.parquet"
MODEL_PATH = ROOT / "artifacts" / "models" / "ranking_model_embeddings.keras"

tf.keras.utils.set_random_seed(42)


def main():
    print("Loading embedding features...")
    df = pd.read_parquet(DATA_PATH)

    feature_columns = [
        col for col in df.columns
        if col.startswith("user_emb_")
        or col.startswith("movie_emb_")
        or col.startswith("genre_")
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)

    print(f"Dataset: {X.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=2,
            restore_best_weights=True,
        )
    ]

    print("Training embedding-based ranker...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=2048,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X_val, batch_size=4096, verbose=0).ravel()

    print("\n--- Classification Validation ---")
    print(f"ROC-AUC:  {roc_auc_score(y_val, probabilities):.4f}")
    print(f"Accuracy: {accuracy_score(y_val, probabilities >= 0.5):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
