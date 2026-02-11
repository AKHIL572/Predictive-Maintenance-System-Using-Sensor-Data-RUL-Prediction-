import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_PATH / "dataset" / "train_FD001.csv"


def load_data():
    """
    Loads the predictive maintenance dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df
