import joblib
import numpy as np
import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data
from src.preprocessor import preprocess_features, add_rul

PROJECT_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_PATH / "models"


def train_model():
    MODEL_PATH.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_data()

    # ── Engine-wise train/test split ──────────────────────────────────────
    # Split on engine IDs, not rows, to prevent data leakage.
    # Rows from the same engine must never appear in both train and test.
    engine_ids = df["engine_id"].unique()
    train_engines, test_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["engine_id"].isin(train_engines)].reset_index(drop=True)
    test_df = df[df["engine_id"].isin(test_engines)].reset_index(drop=True)

    print(
        f"Train engines: {len(train_engines)}  |  Test engines: {len(test_engines)}")

    # ── Training preprocessing ────────────────────────────────────────────
    # preprocess_features computes RUL, applies capping, and derives feature_cols.
    X_train, y_train, feature_cols = preprocess_features(
        train_df, training=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # ── Test set construction ─────────────────────────────────────────────
    # Evaluate against TRUE (uncapped) RUL so test metrics reflect real-world
    # performance, not the training-time approximation.
    # add_rul() is called explicitly here — we do NOT rely on the CSV already
    # having a RUL column, so this works on fresh raw data too.
    test_df = add_rul(test_df)          # computes uncapped ground-truth RUL
    y_test = test_df["RUL"]
    X_test = test_df[feature_cols]

    # ── Cross-validation (GroupKFold) ─────────────────────────────────────
    # GroupKFold ensures no engine's data leaks across folds.
    # n_jobs=1 on the RF so cross_val_score can own the parallelism (n_jobs=-1).
    # Mixing n_jobs=-1 at both levels causes joblib conflicts on Windows.
    groups = train_df["engine_id"].values
    cv = GroupKFold(n_splits=5)

    model_for_cv = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=1,        # let cross_val_score handle parallelism
    )

    scores = cross_val_score(
        model_for_cv,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        groups=groups,
        n_jobs=-1,       # cross_val_score parallelises across folds
    )
    print(
        f"\nCV RMSE (GroupKFold, 5-fold): {-scores.mean():.2f} ± {scores.std():.2f}")

    # ── Final fit on full training set ────────────────────────────────────
    # Use n_jobs=-1 here — no nested parallelism conflict during a plain .fit()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluation on held-out test engines ───────────────────────────────
    preds = model.predict(X_test)

    print("\nHeld-out Test Performance (vs true uncapped RUL):")
    print("  MAE :", round(mean_absolute_error(y_test, preds), 2))
    print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 2))
    print("  R²  :", round(r2_score(y_test, preds), 4))

    # ── Save model and feature list ───────────────────────────────────────
    model_file = MODEL_PATH / "rf_rul_model.pkl"
    features_file = MODEL_PATH / "feature_columns.pkl"

    joblib.dump(model,        model_file)
    joblib.dump(feature_cols, features_file)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n✅ Model saved      → {model_file}")
    print(f"✅ Features saved   → {features_file}")
    print(f"✅ Saved at           {timestamp}")
    print(f"✅ Feature columns  → {feature_cols}")


if __name__ == "__main__":
    train_model()
