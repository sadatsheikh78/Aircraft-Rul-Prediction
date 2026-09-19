import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Baseline XGBoost",
        "Temporal XGBoost",
        "LSTM"
    ],

    "MAE": [
        23.957,
        24.078,
        17.058
    ],

    "RMSE": [
        31.542,
        32.385,
        24.897
    ]
})


print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# IMPROVEMENT
# ============================================================

baseline_mae = 23.957
baseline_rmse = 31.542

lstm_mae = 17.058
lstm_rmse = 24.897

mae_improvement = (
    (baseline_mae - lstm_mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - lstm_rmse)
    / baseline_rmse
) * 100


print("\nLSTM IMPROVEMENT OVER BASELINE")
print("-" * 60)

print(f"MAE improvement  : {mae_improvement:.2f}%")
print(f"RMSE improvement : {rmse_improvement:.2f}%")


# ============================================================
# SAVE TABLE
# ============================================================

results.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# MAE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["MAE"]
)

plt.ylabel("MAE")
plt.title("RUL Prediction Model Comparison - MAE")

plt.tight_layout()

plt.savefig(
    "results/model_comparison_mae.png",
    dpi=300
)

plt.close()


# ============================================================
# RMSE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.ylabel("RMSE")
plt.title("RUL Prediction Model Comparison - RMSE")

plt.tight_layout()

plt.savefig(
    "results/model_comparison_rmse.png",
    dpi=300
)

plt.close()


print("\nFiles saved:")
print("results/model_comparison.csv")
print("results/model_comparison_mae.png")
print("results/model_comparison_rmse.png")