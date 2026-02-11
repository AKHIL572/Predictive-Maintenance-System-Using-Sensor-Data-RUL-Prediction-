import pandas as pd
import joblib
from pathlib import Path

PROJECT_PATH = Path.cwd()
MODEL_PATH = PROJECT_PATH / "Models" / "rf_rul_model.pkl"
FEATURE_PATH = PROJECT_PATH / "Models" / "feature_columns.pkl"
DATA_PATH = PROJECT_PATH / "dataset" / "train_FD001.csv"


def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


def predict_rul(df):
    model, features = load_model()
    X = df[features]
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    # 🔹 Test prediction using sample data
    df = pd.read_csv(DATA_PATH)
    sample = df[df["engine_id"] == 1].tail(1)

    preds = predict_rul(sample)
    print("Predicted RUL:", int(preds[0]))
