import pandas as pd
import joblib
from pathlib import Path

from src.preprocessor import preprocess_features

PROJECT_PATH = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_PATH / "models" / "rf_rul_model.pkl"
FEATURE_FILE = PROJECT_PATH / "models" / "feature_columns.pkl"
DATA_PATH = PROJECT_PATH / "data" / "train_FD001.csv"


def load_model():
    """Load trained model and feature list from disk."""
    if not MODEL_FILE.exists() or not FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Model files not found. Run train.py first to generate them."
        )
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_FILE)
    return model, feature_cols


def predict_rul(df: pd.DataFrame, model=None, feature_cols=None):
    """
    Predict Remaining Useful Life for every row in df.

    Parameters
    ----------
    df : DataFrame with the same raw sensor schema used during training.
         Must include all feature columns saved in feature_columns.pkl.

    Returns
    -------
    numpy array of predicted RUL values (one per row).
    """
    # Accept pre-loaded model/feature_cols to avoid re-deserialising on every call.
    # Falls back to loading from disk if not provided (safe for standalone scripts).
    if model is None or feature_cols is None:
        model, feature_cols = load_model()

    X, _ = preprocess_features(df, training=False, feature_cols=feature_cols)
    return model.predict(X)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    sample = df[df["engine_id"] == 1].tail(1)
    preds = predict_rul(sample)
    print("Predicted RUL:", int(preds[0]))
