import pandas as pd

# Default RUL cap used during training.
# Exposing it as a module-level constant so train.py and tests can import it directly.
RUL_CAP = 125

# op_setting_3 is confirmed constant (always 100.0) in FD001.
# Dropped explicitly by name — NOT via a variance threshold — so op_setting_1
# and op_setting_2 are never silently removed.
CONSTANT_OP_COLS = ["op_setting_3"]


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Remaining Useful Life (RUL) column if not already present.
    RUL = max_cycle_for_engine - current_cycle

    Requires 'engine_id' and 'cycle' columns.
    The last cycle for each engine will have RUL = 0 (run-to-failure).
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


def apply_rul_capping(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Caps RUL at `cap` cycles.

    Only used during training — never at inference.
    Capping reduces the influence of early cycles where no degradation has
    started yet, and aligns the model with the region of interest (near failure).

    Parameters
    ----------
    df  : DataFrame that must contain a 'RUL' column.
    cap : Maximum RUL value. Defaults to module-level RUL_CAP (125).
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def get_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> list:
    """
    Returns sensor column names whose std falls below `threshold`.
    These carry no discriminative signal and are dropped before training.

    Parameters
    ----------
    df        : DataFrame containing sensor_* columns.
    threshold : Std threshold below which a sensor is considered dead.
    """
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]
    sensor_std = df[sensor_cols].std()
    return sensor_std[sensor_std < threshold].index.tolist()


def preprocess_features(
    df: pd.DataFrame,
    training: bool = True,
    feature_cols: list = None,
    rul_cap: int = RUL_CAP,
):
    """
    Full preprocessing pipeline for training and inference.

    Parameters
    ----------
    df           : Raw DataFrame with engine_id, cycle, sensor_* columns.
    training     : If True, derives and returns feature_cols from data.
                   If False, feature_cols must be supplied (loaded from disk).
    feature_cols : Required when training=False. Must be the exact list
                   saved alongside the model in feature_columns.pkl.
    rul_cap      : RUL cap applied during training. Exposed so callers can
                   override without touching the module constant.

    Returns
    -------
    training=True  → (X: DataFrame, y: Series, feature_cols: list)
    training=False → (X: DataFrame, feature_cols: list)
    """
    df = df.copy()

    if training:
        # Step 1: Compute RUL from cycle data (idempotent if already present)
        df = add_rul(df)

        # Step 2: Cap RUL — training only, never at inference
        df = apply_rul_capping(df, cap=rul_cap)

        # Step 3: Identify and drop low-signal sensor columns
        low_variance_sensors = get_low_variance_sensors(df)

        # Step 4: Drop op_setting_3 explicitly (confirmed constant in FD001).
        # op_setting_1 and op_setting_2 are intentionally KEPT — they have
        # genuine (small) variance and should not be removed by a blanket threshold.
        cols_to_exclude = (
            ["engine_id", "cycle", "RUL"]
            + low_variance_sensors
            + CONSTANT_OP_COLS
        )

        feature_cols = [c for c in df.columns if c not in cols_to_exclude]

        X = df[feature_cols]
        y = df["RUL"]
        return X, y, feature_cols

    else:
        # ── Inference mode ────────────────────────────────────────────────
        # Never recompute feature_cols at inference — always use the list
        # that was saved alongside the model to guarantee column alignment.
        if feature_cols is None:
            raise ValueError(
                "feature_cols must be provided when training=False. "
                "Load it from 'feature_columns.pkl' and pass it here."
            )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following expected feature columns are missing from "
                f"the input DataFrame: {missing}"
            )

        X = df[feature_cols]
        return X, feature_cols
