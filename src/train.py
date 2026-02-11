import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data
from src.preprocessor import preprocess_features

PROJECT_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_PATH / "models"
MODEL_PATH.mkdir(exist_ok=True)


def train_model():
    df = load_data()
    X, y, feature_cols = preprocess_features(df, training=True)

    engine_ids = df["engine_id"].unique()
    train_engines, test_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["engine_id"].isin(train_engines)]
    test_df = df[df["engine_id"].isin(test_engines)]

    X_train, y_train, _ = preprocess_features(train_df, training=True)
    X_test, y_test, _ = preprocess_features(test_df, training=True)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Model Performance")
    print("MAE :", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2  :", r2_score(y_test, preds))

    joblib.dump(model, MODEL_PATH / "rf_rul_model.pkl")
    joblib.dump(feature_cols, MODEL_PATH / "feature_columns.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()
