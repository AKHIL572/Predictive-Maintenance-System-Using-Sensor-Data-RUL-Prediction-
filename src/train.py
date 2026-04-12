import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data
from src.preprocessor import preprocess_features, add_rul, apply_rul_capping

PROJECT_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_PATH / "models"


def train_model():
    # Fix 15: mkdir runs only when actually training, not at import time
    MODEL_PATH.mkdir(exist_ok=True)

    df = load_data()

    # ── Engine-wise split (prevents data leakage) ───────────────────────────
    engine_ids = df["engine_id"].unique()

    train_engines, test_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    # Fix 5: reset_index so that X_train, y_train, and groups share a clean 0-based index
    train_df = df[df["engine_id"].isin(train_engines)].reset_index(drop=True)
    test_df  = df[df["engine_id"].isin(test_engines)].reset_index(drop=True)

    # ── Training-set preprocessing → derives feature_cols ──────────────────
    X_train, y_train, feature_cols = preprocess_features(train_df, training=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Fix 4: test set uses the SAME feature_cols — no independent recomputation
    test_processed = apply_rul_capping(add_rul(test_df))
    X_test = test_processed[feature_cols]
    y_test = test_processed["RUL"]

    # Fix 5: groups aligned to X_train's index
    groups = train_df["engine_id"].reset_index(drop=True)

    # ── Model definition ────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    # Fix 11: CV runs BEFORE final fit so scores reflect generalisation, not the fitted object
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        groups=groups,
    )
    print(f"CV RMSE (GroupKFold, 5-fold): {-scores.mean():.2f} ± {scores.std():.2f}")

    # ── Final fit on full training set ──────────────────────────────────────
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nHeld-out Test Performance:")
    print("MAE :", round(mean_absolute_error(y_test, preds), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 2))
    print("R2  :", round(r2_score(y_test, preds), 4))

    # ── Save artefacts ──────────────────────────────────────────────────────
    joblib.dump(model,        MODEL_PATH / "rf_rul_model.pkl")
    joblib.dump(feature_cols, MODEL_PATH / "feature_columns.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train_model()