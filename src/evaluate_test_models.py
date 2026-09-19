import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_PATH = "data/processed/test_fd001_with_rul.csv"

XGB_MODEL_PATH = "models/xgboost_rul_baseline.joblib"

LSTM_MODEL_PATH = "models/lstm_rul_model.keras"

LSTM_SCALER_PATH = "models/lstm_scaler.joblib"

XGB_PREDICTIONS_PATH = (
    "results/xgboost_official_test_predictions.csv"
)

LSTM_PREDICTIONS_PATH = (
    "results/lstm_official_test_predictions.csv"
)

COMPARISON_PATH = (
    "results/official_test_model_comparison.csv"
)

WINDOW_SIZE = 30


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("OFFICIAL C-MAPSS FD001 TEST EVALUATION")
print("=" * 60)

df = pd.read_csv(TEST_DATA_PATH)

df = df.sort_values(
    ["unit_id", "cycle"]
).reset_index(drop=True)

print(f"Test rows      : {len(df)}")
print(f"Test engines   : {df['unit_id'].nunique()}")
print(f"Test columns   : {len(df.columns)}")


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

possible_target_columns = [
    "RUL",
    "rul",
    "actual_rul"
]

target_column = None

for column in possible_target_columns:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    raise ValueError(
        "Could not find the RUL target column. "
        f"Available columns: {list(df.columns)}"
    )

print(
    f"\nTarget column detected: {target_column}"
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    col
    for col in df.columns
    if col not in [
        "unit_id",
        "cycle",
        target_column
    ]
    and df[col].nunique() <= 1
]

print("\nRemoving constant features:")
print(constant_features)


# ============================================================
# MODEL FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col not in [
        "unit_id",
        target_column
    ]
    and col not in constant_features
]

print(
    f"\nNumber of model features: "
    f"{len(feature_columns)}"
)


# ============================================================
# TARGET
# ============================================================

y_test = df[target_column].values


# ============================================================
# XGBOOST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST OFFICIAL TEST EVALUATION")
print("=" * 60)

X_test_xgb = df[
    feature_columns
].values

xgb_model = joblib.load(
    XGB_MODEL_PATH
)

xgb_predictions = xgb_model.predict(
    X_test_xgb
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

print(
    f"MAE  : {xgb_mae:.3f}"
)

print(
    f"RMSE : {xgb_rmse:.3f}"
)


# ============================================================
# SAVE XGBOOST PREDICTIONS
# ============================================================

xgb_prediction_df = pd.DataFrame({
    "unit_id": df["unit_id"].values,
    "cycle": df["cycle"].values,
    "actual_rul": y_test,
    "predicted_rul": xgb_predictions
})

xgb_prediction_df.to_csv(
    XGB_PREDICTIONS_PATH,
    index=False
)


# ============================================================
# PREPARE LSTM TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPARING LSTM TEST DATA")
print("=" * 60)

print("\nLoading training scaler...")

scaler = joblib.load(
    LSTM_SCALER_PATH
)

print(
    f"Scaler loaded: {LSTM_SCALER_PATH}"
)


# ------------------------------------------------------------
# Preserve original metadata before scaling
# ------------------------------------------------------------

metadata = pd.DataFrame({
    "unit_id": df["unit_id"].values,
    "cycle": df["cycle"].values,
    "actual_rul": y_test
})


# ------------------------------------------------------------
# Extract ONLY model features
# ------------------------------------------------------------

lstm_features = df[
    feature_columns
].copy()


# ------------------------------------------------------------
# Apply training-fitted scaler
# ------------------------------------------------------------

lstm_features_scaled = scaler.transform(
    lstm_features
)

print(
    "Training scaler successfully "
    "applied to test data."
)


# ============================================================
# CREATE LSTM TEST SEQUENCES
# ============================================================

print("\nCreating LSTM test sequences...")

X_test_sequences = []
y_test_sequences = []

sequence_metadata = []


for unit_id in df["unit_id"].unique():

    unit_mask = (
        df["unit_id"] == unit_id
    )

    unit_features = (
        lstm_features_scaled[
            unit_mask.values
        ]
    )

    unit_metadata = (
        metadata[unit_mask]
        .reset_index(drop=True)
    )

    for i in range(
        WINDOW_SIZE,
        len(unit_features) + 1
    ):

        start = i - WINDOW_SIZE
        end = i

        X_test_sequences.append(
            unit_features[start:end]
        )

        y_test_sequences.append(
            unit_metadata.iloc[i - 1][
                "actual_rul"
            ]
        )

        sequence_metadata.append({
            "unit_id":
                int(
                    unit_metadata.iloc[i - 1][
                        "unit_id"
                    ]
                ),

            "cycle":
                int(
                    unit_metadata.iloc[i - 1][
                        "cycle"
                    ]
                ),

            "actual_rul":
                float(
                    unit_metadata.iloc[i - 1][
                        "actual_rul"
                    ]
                )
        })


X_test_sequences = np.array(
    X_test_sequences
)

y_test_sequences = np.array(
    y_test_sequences
)


print(
    f"X_test shape : "
    f"{X_test_sequences.shape}"
)

print(
    f"y_test shape : "
    f"{y_test_sequences.shape}"
)


# ============================================================
# LOAD LSTM MODEL
# ============================================================

print("\nLoading LSTM model...")

lstm_model = load_model(
    LSTM_MODEL_PATH
)


# ============================================================
# LSTM PREDICTIONS
# ============================================================

print("Generating LSTM predictions...")

lstm_predictions = (
    lstm_model.predict(
        X_test_sequences,
        verbose=0
    )
    .reshape(-1)
)


# ============================================================
# LSTM METRICS
# ============================================================

lstm_mae = mean_absolute_error(
    y_test_sequences,
    lstm_predictions
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_sequences,
        lstm_predictions
    )
)


print("\n" + "=" * 60)
print("LSTM OFFICIAL TEST PERFORMANCE")
print("=" * 60)

print(
    f"MAE  : {lstm_mae:.3f}"
)

print(
    f"RMSE : {lstm_rmse:.3f}"
)


# ============================================================
# SAVE LSTM PREDICTIONS
# ============================================================

lstm_prediction_df = pd.DataFrame(
    sequence_metadata
)

lstm_prediction_df[
    "predicted_rul"
] = lstm_predictions

lstm_prediction_df.to_csv(
    LSTM_PREDICTIONS_PATH,
    index=False
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("OFFICIAL TEST MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Baseline XGBoost",
        "LSTM"
    ],
    "MAE": [
        xgb_mae,
        lstm_mae
    ],
    "RMSE": [
        xgb_rmse,
        lstm_rmse
    ]
})

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nFiles saved:")
print(XGB_PREDICTIONS_PATH)
print(LSTM_PREDICTIONS_PATH)
print(COMPARISON_PATH)

print(
    "\nOfficial test evaluation "
    "completed successfully."
)