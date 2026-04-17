import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_PATH / "data" / "train_FD001.csv"


def load_data(path: Path = None) -> pd.DataFrame:
    """
    Loads the dataset from disk.

    Parameters
    ----------
    path : Optional path to a CSV file. Defaults to train_FD001.csv.

    Returns
    -------
    pd.DataFrame
    """
    data_path = Path(path) if path else DEFAULT_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df
