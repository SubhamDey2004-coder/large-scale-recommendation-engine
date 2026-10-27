
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"

RANDOM_SEED = 42
BATCH_SIZE = 4096
EPOCHS = 5


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_DIR / "ranking_train.parquet")

    genre_columns = [
        column for column in df.columns
        if column.startswith("genre_")
    ]

    # Use IDs as categorical features and genres as content features.
    user_ids = df["userId"].to_numpy(dtype=np.int32)
    movie_ids = df["movieId"].to_numpy(dtype=np.int32)
    genres = df[genre_columns].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.float32)

    user_scaler = StandardScaler()
    movie_scaler = StandardScaler()

    user_features = user_scaler.fit_transform(
        user_ids.reshape(-1, 1)
    ).astype(np.float32)

    movie_features = movie_scaler.fit_transform(
        movie_ids.reshape(-1, 1)
    ).astype(np.float32)

    X = np.column_stack([
        user_features,
        movie_features,
        genres
    ]).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=2,
                mode="max",
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    model.save(MODEL_DIR / "ranking_model.keras")
    joblib.dump(user_scaler, MODEL_DIR / "user_scaler.joblib")
    joblib.dump(movie_scaler, MODEL_DIR / "movie_scaler.joblib")
    np.save(MODEL_DIR / "ranking_genre_columns.npy", genre_columns)

    results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)

    print("\nValidation results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nModel saved to: {MODEL_DIR / 'ranking_model.keras'}")


if __name__ == "__main__":
    main()
