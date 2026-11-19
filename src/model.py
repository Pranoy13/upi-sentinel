
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.features import build_features, FEATURE_COLS


MODEL_PATH = os.path.join("models", "upi_sentinel_model.joblib")

# Trained model objects
_model_bundle = None


def _prepare_features(df):
    """Build features if needed and return only approved ML columns."""

    if not set(FEATURE_COLS).issubset(df.columns):
        df = build_features(df.copy())

    X = df[FEATURE_COLS].copy()

    # Ensure all model inputs are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def train(df):
    """
    Train Isolation Forest and Random Forest.

    Uses chronological 70% training and 30% holdout.
    is_fraud is used ONLY as the Random Forest target.
    scenario is never used as a feature.
    """

    global _model_bundle

    if "is_fraud" not in df.columns:
        raise ValueError("Training data must contain is_fraud labels.")

    if "timestamp" not in df.columns:
        raise ValueError("Training data must contain timestamp.")

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    X = _prepare_features(data)
    y = data["is_fraud"].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")

    split_index = int(len(data) * 0.70)

    if split_index == 0 or split_index >= len(data):
        raise ValueError("Not enough data for a 70:30 split.")

    X_train = X.iloc[:split_index]
    X_holdout = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_holdout = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError("Training period must contain both classes.")

    # Fit imputation using training data only.
    imputer = SimpleImputer(strategy="median")

    X_train_clean = imputer.fit_transform(X_train)

    # 1. Unsupervised anomaly detection.
    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    isolation_forest.fit(X_train_clean)

    # 2. Supervised fraud classification.
    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    random_forest.fit(X_train_clean, y_train)

    # Save both models and preprocessing object.
    _model_bundle = {
        "imputer": imputer,
        "isolation_forest": isolation_forest,
        "random_forest": random_forest,
        "feature_cols": FEATURE_COLS,
        "train_end_timestamp": str(data.iloc[split_index - 1]["timestamp"]),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(_model_bundle, MODEL_PATH)

    print("\nMODEL TRAINING COMPLETE")
    print("-----------------------")
    print(f"Total transactions: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Holdout rows: {len(X_holdout)}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Holdout fraud cases: {int(y_holdout.sum())}")
    print(f"Saved model: {MODEL_PATH}")

    return {
        "total_rows": len(data),
        "train_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "train_fraud": int(y_train.sum()),
        "holdout_fraud": int(y_holdout.sum()),
        "model_path": MODEL_PATH,
    }


def predict_proba(df):
    """
    Generate ML risk outputs.

    Returns a DataFrame containing:
        rf_prob: Random Forest fraud probability.
        anomaly_score: Isolation Forest anomaly score.
        model_prob: Combined ML probability (0 to 1).
    """

    global _model_bundle

    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            if "is_fraud" in df.columns:
                print("No saved model found. Training one now (first run only)...")
                train(df)
            else:
                raise FileNotFoundError(
                    "Model not found. Run train(df) first."
                )

        if _model_bundle is None:
            _model_bundle = joblib.load(MODEL_PATH)

    X = _prepare_features(df)

    X_clean = _model_bundle["imputer"].transform(X)

    rf = _model_bundle["random_forest"]
    iso = _model_bundle["isolation_forest"]

    # Random Forest fraud probability.
    fraud_class_index = list(rf.classes_).index(1)
    rf_prob = rf.predict_proba(X_clean)[:, fraud_class_index]

    # Convert Isolation Forest anomaly score to 0-1.
    raw_anomaly = iso.decision_function(X_clean)

    anomaly_score = 1.0 / (
        1.0 + np.exp(np.clip(raw_anomaly * 10, -50, 50))
    )

    # Combined model probability.
    model_prob = (
        0.75 * rf_prob +
        0.25 * anomaly_score
    )

    return pd.DataFrame(
        {
            "rf_prob": rf_prob,
            "anomaly_score": anomaly_score,
            "model_prob": model_prob,
        },
        index=df.index,
    )
