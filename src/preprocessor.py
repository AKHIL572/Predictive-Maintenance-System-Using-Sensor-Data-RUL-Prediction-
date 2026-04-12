import pandas as pd

RUL_CAP = 125


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Remaining Useful Life (RUL) column if not already present.
    RUL = max_cycle_for_engine - current_cycle
    """
    df = df.copy()

    if "RUL" not in df.columns:
        max_cycle = df.groupby("engine_id")["cycle"].transform("max")
        df["RUL"] = max_cycle - df["cycle"]

    return df


def apply_rul_capping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps RUL at RUL_CAP for training stability.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def get_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> list:
    """
    Returns sensor column names whose standard deviation falls below `threshold`.
    These sensors carry little signal and are dropped before training.
    """
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]
    sensor_std = df[sensor_cols].std()                    # std (not var) — threshold is for std
    return sensor_std[sensor_std < threshold].index.tolist()


def preprocess_features(
    df: pd.DataFrame,
    training: bool = True,
    feature_cols: list = None,
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : Raw DataFrame (must contain engine_id, cycle, sensor_* columns).
    training     : If True, compute and return feature_cols from the data.
                   If False, feature_cols MUST be supplied (e.g. loaded from .pkl).
    feature_cols : List of column names to select as model features.
                   Required when training=False. Ignored when training=True.

    Returns
    -------
    training=True  → (X: DataFrame, y: Series, feature_cols: list)
    training=False → (X: DataFrame, feature_cols: list)
    """
    df = df.copy()

    # Step 1: Ensure RUL column exists
    df = add_rul(df)

    # Step 2: Cap RUL
    df = apply_rul_capping(df)

    if training:
        # Derive feature_cols from THIS dataset (training data only)
        low_variance_sensors = get_low_variance_sensors(df)
        feature_cols = df.drop(
            columns=["engine_id", "cycle", "RUL"] + low_variance_sensors,
            errors="ignore",
        ).columns.tolist()

        X = df[feature_cols]
        y = df["RUL"]
        return X, y, feature_cols

    else:
        # Inference mode: NEVER recompute feature_cols from input data.
        # A single row would have std=0 for all sensors, dropping everything.
        if feature_cols is None:
            raise ValueError(
                "feature_cols must be provided when training=False. "
                "Pass the list loaded from 'feature_columns.pkl'."
            )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following expected features are missing from the input DataFrame: {missing}"
            )

        X = df[feature_cols]
        return X, feature_cols