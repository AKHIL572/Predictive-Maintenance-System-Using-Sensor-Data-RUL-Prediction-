import pandas as pd

DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19"
]


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Remaining Useful Life (RUL) column.
    """
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def preprocess_features(df: pd.DataFrame, training=True):
    """
    Cleans dataset and returns features (and target if training).
    """
    df = df.copy()

    if training:
        df = add_rul(df)

    feature_cols = df.drop(
        columns=["engine_id", "cycle", "RUL"] + DROP_SENSORS,
        errors="ignore"
    ).columns.tolist()

    X = df[feature_cols]

    if training:
        y = df["RUL"]
        return X, y, feature_cols

    return X
