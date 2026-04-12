import pandas as pd
import joblib
from pathlib import Path

from src.preprocessor import preprocess_features

PROJECT_PATH = Path(__file__).resolve().parents[1]

MODEL_PATH   = PROJECT_PATH / "models" / "rf_rul_model.pkl"
FEATURE_PATH = PROJECT_PATH / "models" / "feature_columns.pkl"
DATA_PATH    = PROJECT_PATH / "data"   / "train_FD001.csv"


def load_model():
    """Load the trained model and feature column list from disk."""
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols


# Fix 7: Load model ONCE at module level — not on every predict_rul() call
_model, _feature_cols = load_model()


def predict_rul(df: pd.DataFrame):
    """
    Predict Remaining Useful Life for every row in df.

    Parameters
    ----------
    df : DataFrame with the same raw sensor schema used during training.

    Returns
    -------
    numpy array of predicted RUL values (one per row).
    """
    # Pass feature_cols explicitly to avoid re-deriving them from a small input (Bug 6)
    X, _ = preprocess_features(df, training=False, feature_cols=_feature_cols)
    return _model.predict(X)


if __name__ == "__main__":
    df     = pd.read_csv(DATA_PATH)
    sample = df[df["engine_id"] == 1].tail(1)
    preds  = predict_rul(sample)
    print("Predicted RUL:", int(preds[0]))