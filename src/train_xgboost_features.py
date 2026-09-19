from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "train_fd001_features.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("TIME-SERIES FEATURE-ENGINEERED XGBOOST")
    print("=" * 60)

    df = pd.read_csv(DATA)

    # --------------------------------------------------
    # Remove identifiers and target
    # --------------------------------------------------

    features = [
        col
        for col in df.columns
        if col not in ["unit_id", "rul"]
    ]

    X = df[features]
    y = df["rul"]

    # --------------------------------------------------
    # Split by ENGINE
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=df["unit_id"]
        )
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(f"\nTotal features     : {len(features)}")
    print(f"Training rows      : {len(X_train)}")
    print(f"Validation rows    : {len(X_val)}")
    print(
        f"Training engines   : "
        f"{df.iloc[train_idx]['unit_id'].nunique()}"
    )
    print(
        f"Validation engines : "
        f"{df.iloc[val_idx]['unit_id'].nunique()}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = model.predict(X_val)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = mean_squared_error(
        y_val,
        predictions
    ) ** 0.5

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # --------------------------------------------------
    # Compare with baseline
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    print("Baseline MAE  : 23.957")
    print("Baseline RMSE : 31.542")

    print(f"Feature MAE   : {mae:.3f}")
    print(f"Feature RMSE  : {rmse:.3f}")

    mae_change = (
        (23.957 - mae) / 23.957
    ) * 100

    rmse_change = (
        (31.542 - rmse) / 31.542
    ) * 100

    print(
        f"\nMAE improvement : {mae_change:.2f}%"
    )

    print(
        f"RMSE improvement: {rmse_change:.2f}%"
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        MODELS /
        "xgboost_temporal_features.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    results = df.iloc[val_idx][
        ["unit_id", "cycle", "rul"]
    ].copy()

    results["predicted_rul"] = predictions

    results.to_csv(
        RESULTS /
        "xgboost_temporal_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        RESULTS /
        "xgboost_temporal_feature_importance.csv",
        index=False
    )

    print("\nTop 15 features:")

    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    print("\nModel saved:")
    print(model_path)


if __name__ == "__main__":
    main()