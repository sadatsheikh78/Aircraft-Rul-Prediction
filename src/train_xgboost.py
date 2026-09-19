from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "train_fd001_with_rul.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def main():

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    df = pd.read_csv(DATA)

    print("=" * 60)
    print("XGBOOST RUL BASELINE")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------
    # 2. REMOVE CONSTANT FEATURES
    # --------------------------------------------------

    excluded_columns = ["unit_id", "rul"]

    candidate_features = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    constant_features = [
        col
        for col in candidate_features
        if df[col].nunique() <= 1
    ]

    features = [
        col
        for col in candidate_features
        if col not in constant_features
    ]

    print("\nRemoved constant features:")
    print(constant_features)

    print(f"\nNumber of model features: {len(features)}")


    # --------------------------------------------------
    # 3. GROUP-BASED TRAIN/VALIDATION SPLIT
    # --------------------------------------------------

    # Important:
    # Rows from the same engine should not be randomly
    # scattered across both training and validation sets.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["unit_id"]
        )
    )

    train = df.iloc[train_idx]
    validation = df.iloc[val_idx]


    print("\nTraining engines   :", train["unit_id"].nunique())
    print("Validation engines :", validation["unit_id"].nunique())


    # --------------------------------------------------
    # 4. CREATE X AND y
    # --------------------------------------------------

    X_train = train[features]
    y_train = train["rul"]

    X_val = validation[features]
    y_val = validation["rul"]


    # --------------------------------------------------
    # 5. XGBOOST MODEL
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
    # 6. PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X_val)


    # --------------------------------------------------
    # 7. EVALUATION
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
    # 8. SAVE MODEL
    # --------------------------------------------------

    model_path = (
        MODELS /
        "xgboost_rul_baseline.joblib"
    )

    joblib.dump(
        model,
        model_path
    )


    # --------------------------------------------------
    # 9. SAVE PREDICTIONS
    # --------------------------------------------------

    results = validation[
        ["unit_id", "cycle", "rul"]
    ].copy()

    results["predicted_rul"] = predictions

    results.to_csv(
        RESULTS /
        "xgboost_validation_predictions.csv",
        index=False
    )


    # --------------------------------------------------
    # 10. FEATURE IMPORTANCE
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
        "xgboost_feature_importance.csv",
        index=False
    )


    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    print("\nSaved model:")
    print(model_path)

    print("\nBaseline completed successfully.")


if __name__ == "__main__":
    main()