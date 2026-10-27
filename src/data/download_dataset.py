from pathlib import Path
import zipfile

import truststore
truststore.inject_into_ssl()

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / "ml-25m.zip"
EXTRACT_DIR = DATA_DIR / "ml-25m"

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if (EXTRACT_DIR / "ratings.csv").exists():
        print("MovieLens 25M is already extracted.")
        return

    if not ZIP_PATH.exists():
        print("Downloading MovieLens 25M...")
        response = requests.get(
            DATASET_URL, stream=True, timeout=(20, 120)
        )
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with ZIP_PATH.open("wb") as file:
            with tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                desc="Download",
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))

    print("Checking ZIP archive...")
    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP archive: {ZIP_PATH}"
        )

    print("Extracting dataset...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    ratings_path = EXTRACT_DIR / "ratings.csv"
    movies_path = EXTRACT_DIR / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise RuntimeError("Dataset extraction is incomplete.")

    print("\nDataset downloaded and extracted successfully.")
    print(f"Ratings: {ratings_path}")
    print(f"Movies:  {movies_path}")


def inspect_dataset() -> None:
    ratings_path = EXTRACT_DIR / "ratings.csv"
    movies_path = EXTRACT_DIR / "movies.csv"

    ratings = pd.read_csv(ratings_path, nrows=5)
    movies = pd.read_csv(movies_path, nrows=5)

    print("\nRatings preview:")
    print(ratings.to_string(index=False))
    print("\nRatings columns:", ratings.columns.tolist())
    print("Movies columns:", movies.columns.tolist())


if __name__ == "__main__":
    download_dataset()
    inspect_dataset()

