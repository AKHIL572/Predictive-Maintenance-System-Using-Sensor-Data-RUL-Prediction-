import pandas as pd

RUL_CAP = 125


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Remaining Useful Life (RUL) column if not already present.
    RUL = max_cycle_for_engine - current_cycle
    Requires 'engine_id' and 'cycle' columns.
    """
    df = df.copy()

    if "RUL" not in df.columns:
        if "engine_id" not in df.columns or "cycle" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'engine_id' and 'cycle' columns to compute RUL."
            )
        max_cycle = df.groupby("engine_id")["cycle"].transform("max")
        df["RUL"] = max_cycle - df["cycle"]

    return df


def apply_rul_capping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps RUL at RUL_CAP. Only used during training — not at inference.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def get_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> list:
    """
    Returns sensor column names whose std falls below threshold.
    These carry little signal and are dropped before training.
    """
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]
    sensor_std = df[sensor_cols].std()
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
    df           : Raw DataFrame with engine_id, cycle, sensor_* columns.
    training     : If True, derives and returns feature_cols from data.
                   If False, feature_cols must be supplied.
    feature_cols : Required when training=False.

    Returns
    -------
    training=True  → (X: DataFrame, y: Series, feature_cols: list)
    training=False → (X: DataFrame, feature_cols: list)
    """
    df = df.copy()

    if training:
        # Step 1: Add RUL
        df = add_rul(df)

        # Step 2: Cap RUL — only during training
        df = apply_rul_capping(df)

        # Step 3: Derive feature columns from training data
        low_variance_sensors = get_low_variance_sensors(df)
        low_variance_ops = [
            col for col in df.columns
            if col.startswith("op_setting") and df[col].std() < 0.01
        ]
        cols_to_drop = ["engine_id", "cycle", "RUL"] + \
            low_variance_sensors + low_variance_ops

        feature_cols = [c for c in df.columns if c not in cols_to_drop]

        X = df[feature_cols]
        y = df["RUL"]
        return X, y, feature_cols

    else:
        # Inference mode — never recompute feature_cols
        if feature_cols is None:
            raise ValueError(
                "feature_cols must be provided when training=False. "
                "Pass the list loaded from 'feature_columns.pkl'."
            )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following expected features are missing from input: {missing}"
            )

        X = df[feature_cols]
        return X, feature_cols
