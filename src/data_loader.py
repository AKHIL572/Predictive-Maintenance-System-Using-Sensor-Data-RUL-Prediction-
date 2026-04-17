import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_PATH / "data" / "train_FD001.csv"

# Minimum columns that must exist in any valid dataset file
EXPECTED_COLS = {"engine_id", "cycle"}


def load_data(path: Path = None) -> pd.DataFrame:
    """
    Loads the dataset from disk and validates its schema.

    Parameters
    ----------
    path : Optional path to a CSV file. Defaults to train_FD001.csv.

    Returns
    -------
    pd.DataFrame with engine_id and cycle guaranteed to be int dtype.

    Raises
    ------
    FileNotFoundError : if the file does not exist at the given path.
    ValueError        : if required columns are missing from the file.
    """
    data_path = Path(path) if path else DEFAULT_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Schema validation — catch corrupt or wrong files early
    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Loaded CSV is missing required columns: {missing_cols}. "
            f"Make sure you are loading a processed FD001 file."
        )

    # Enforce correct dtypes — raw .txt files can load these as float
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df
