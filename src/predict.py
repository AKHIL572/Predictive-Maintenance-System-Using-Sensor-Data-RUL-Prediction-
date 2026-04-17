import pandas as pd
import joblib
from pathlib import Path

from src.preprocessor import preprocess_features

PROJECT_PATH = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_PATH / "models" / "rf_rul_model.pkl"
FEATURE_FILE = PROJECT_PATH / "models" / "feature_columns.pkl"
DATA_PATH = PROJECT_PATH / "data" / "train_FD001.csv"


def load_model():
    """
    Load trained model and feature column list from disk.

    Returns
    -------
    model        : Trained RandomForestRegressor.
    feature_cols : List of feature column names the model expects.

    Raises
    ------
    FileNotFoundError : if model files have not been generated yet.
    """
    if not MODEL_FILE.exists() or not FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python -m src.train` first to generate them."
        )
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_FILE)
    return model, feature_cols


def predict_rul(
    df: pd.DataFrame,
    model=None,
    feature_cols: list = None,
) -> "np.ndarray":
    """
    Predict Remaining Useful Life for every row in df.

    Parameters
    ----------
    df           : DataFrame with the same raw sensor schema used during training.
                   Must contain all columns listed in feature_columns.pkl.
    model        : Pre-loaded model object. If None, loaded from disk.
    feature_cols : Pre-loaded feature list. If None, loaded from disk.
                   Pass both together to avoid re-deserialising on every call
                   (important for the Streamlit app where caching handles this).

    Returns
    -------
    numpy array of predicted RUL values (one float per row).
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model()

    X, _ = preprocess_features(df, training=False, feature_cols=feature_cols)
    return model.predict(X)


if __name__ == "__main__":
    """
    Quick smoke-test: predict RUL for a mid-life cycle of engine 1.

    Intentionally avoids the last cycle (RUL = 0 at failure) which would
    give a misleadingly low predicted value and look like a bad prediction.
    """
    df = pd.read_csv(DATA_PATH)

    engine_df = df[df["engine_id"] == 1].sort_values(
        "cycle").reset_index(drop=True)

    # Pick the mid-life row — not the first (no degradation yet) and not the
    # last (always RUL = 0, gives a misleading demo).
    mid_idx = len(engine_df) // 2
    sample = engine_df.iloc[[mid_idx]]

    preds = predict_rul(sample)
    actual = int(sample["RUL"].values[0]
                 ) if "RUL" in sample.columns else "unknown"

    print(f"Engine 1 — cycle {int(sample['cycle'].values[0])}")
    print(f"  Predicted RUL : {int(preds[0])}")
    print(f"  Actual RUL    : {actual}")
