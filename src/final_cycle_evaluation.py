import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

XGB_PATH = (
    "results/xgboost_official_test_predictions.csv"
)

LSTM_PATH = (
    "results/lstm_official_test_predictions.csv"
)

OUTPUT_PATH = (
    "results/final_cycle_model_comparison.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 60)
print("FINAL-CYCLE OFFICIAL C-MAPSS FD001 EVALUATION")
print("=" * 60)

xgb = pd.read_csv(XGB_PATH)

lstm = pd.read_csv(LSTM_PATH)

print(
    f"\nXGBoost prediction rows : {len(xgb)}"
)

print(
    f"LSTM prediction rows    : {len(lstm)}"
)


# ============================================================
# SELECT FINAL OBSERVED CYCLE
# ============================================================

xgb_final = (
    xgb.sort_values(
        ["unit_id", "cycle"]
    )
    .groupby(
        "unit_id",
        as_index=False
    )
    .tail(1)
    .reset_index(drop=True)
)

lstm_final = (
    lstm.sort_values(
        ["unit_id", "cycle"]
    )
    .groupby(
        "unit_id",
        as_index=False
    )
    .tail(1)
    .reset_index(drop=True)
)


# ============================================================
# CHECK ENGINE COUNTS
# ============================================================

print(
    f"\nXGBoost final engines : "
    f"{xgb_final['unit_id'].nunique()}"
)

print(
    f"LSTM final engines    : "
    f"{lstm_final['unit_id'].nunique()}"
)


# ============================================================
# XGBOOST METRICS
# ============================================================

xgb_mae = mean_absolute_error(
    xgb_final["actual_rul"],
    xgb_final["predicted_rul"]
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        xgb_final["actual_rul"],
        xgb_final["predicted_rul"]
    )
)


# ============================================================
# LSTM METRICS
# ============================================================

lstm_mae = mean_absolute_error(
    lstm_final["actual_rul"],
    lstm_final["predicted_rul"]
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        lstm_final["actual_rul"],
        lstm_final["predicted_rul"]
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL-CYCLE MODEL PERFORMANCE")
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


# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    (xgb_mae - lstm_mae)
    / xgb_mae
    * 100
)

rmse_improvement = (
    (xgb_rmse - lstm_rmse)
    / xgb_rmse
    * 100
)

print(
    f"\nLSTM MAE improvement  : "
    f"{mae_improvement:.2f}%"
)

print(
    f"LSTM RMSE improvement : "
    f"{rmse_improvement:.2f}%"
)


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

xgb_final.to_csv(
    "results/xgboost_final_cycle_predictions.csv",
    index=False
)

lstm_final.to_csv(
    "results/lstm_final_cycle_predictions.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFiles saved:")

print(
    "results/xgboost_final_cycle_predictions.csv"
)

print(
    "results/lstm_final_cycle_predictions.csv"
)

print(
    OUTPUT_PATH
)

print(
    "\nFinal-cycle evaluation completed successfully."
)