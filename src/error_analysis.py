import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_PATH = "results/lstm_official_test_predictions.csv"

OUTPUT_SUMMARY = "results/lstm_error_analysis.csv"

ERROR_PLOT = "results/lstm_error_distribution.png"

RUL_ERROR_PLOT = "results/lstm_error_by_rul_range.png"

ENGINE_ERROR_PLOT = "results/lstm_error_by_engine.png"

ACTUAL_PREDICTED_PLOT = (
    "results/lstm_official_actual_vs_predicted.png"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 60)
print("LSTM OFFICIAL TEST ERROR ANALYSIS")
print("=" * 60)

df = pd.read_csv(PREDICTION_PATH)

print(f"Prediction rows : {len(df)}")
print(f"Engines         : {df['unit_id'].nunique()}")


# ============================================================
# ERROR CALCULATION
# ============================================================

df["error"] = (
    df["predicted_rul"] - df["actual_rul"]
)

df["absolute_error"] = (
    np.abs(df["error"])
)


# ============================================================
# OVERALL ERROR STATISTICS
# ============================================================

print("\nOVERALL ERROR")
print("-" * 60)

print(
    f"Mean error      : {df['error'].mean():.3f}"
)

print(
    f"Mean abs error  : "
    f"{df['absolute_error'].mean():.3f}"
)

print(
    f"Median abs error: "
    f"{df['absolute_error'].median():.3f}"
)

print(
    f"Max abs error   : "
    f"{df['absolute_error'].max():.3f}"
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

print("\nTOP 10 LARGEST ERRORS")
print("-" * 60)

worst = (
    df.sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)

print(
    worst[
        [
            "unit_id",
            "cycle",
            "actual_rul",
            "predicted_rul",
            "absolute_error"
        ]
    ].to_string(index=False)
)


# ============================================================
# ERROR BY RUL RANGE
# ============================================================

df["rul_range"] = pd.cut(
    df["actual_rul"],
    bins=[
        -np.inf,
        30,
        60,
        100,
        150,
        200,
        np.inf
    ],
    labels=[
        "0-30",
        "31-60",
        "61-100",
        "101-150",
        "151-200",
        "200+"
    ]
)

range_analysis = (
    df.groupby(
        "rul_range",
        observed=False
    )
    .agg(
        samples=("actual_rul", "count"),
        mae=("absolute_error", "mean")
    )
    .reset_index()
)

print("\nERROR BY ACTUAL RUL RANGE")
print("-" * 60)

print(
    range_analysis.to_string(
        index=False
    )
)


# ============================================================
# ERROR BY ENGINE
# ============================================================

engine_analysis = (
    df.groupby("unit_id")
    .agg(
        samples=("actual_rul", "count"),
        mae=("absolute_error", "mean"),
        max_error=("absolute_error", "max")
    )
    .reset_index()
    .sort_values(
        "mae",
        ascending=False
    )
)

print("\nTOP 10 ENGINES BY MAE")
print("-" * 60)

print(
    engine_analysis.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE ERROR SUMMARY
# ============================================================

range_analysis.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED RUL
# ============================================================

plt.figure(figsize=(10, 6))

sample_size = min(
    2000,
    len(df)
)

sample = df.iloc[:sample_size]

plt.scatter(
    sample["actual_rul"],
    sample["predicted_rul"],
    alpha=0.3
)

min_value = min(
    sample["actual_rul"].min(),
    sample["predicted_rul"].min()
)

max_value = max(
    sample["actual_rul"].max(),
    sample["predicted_rul"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title(
    "LSTM Official Test: Actual vs Predicted RUL"
)

plt.tight_layout()

plt.savefig(
    ACTUAL_PREDICTED_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["error"],
    bins=50
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.title(
    "LSTM Prediction Error Distribution"
)

plt.tight_layout()

plt.savefig(
    ERROR_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# ERROR BY RUL RANGE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    range_analysis["rul_range"].astype(str),
    range_analysis["mae"]
)

plt.xlabel("Actual RUL Range")
plt.ylabel("MAE")

plt.title(
    "LSTM Error by Actual RUL Range"
)

plt.tight_layout()

plt.savefig(
    RUL_ERROR_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# ERROR BY ENGINE
# ============================================================

plt.figure(figsize=(12, 6))

top_engines = engine_analysis.head(15)

plt.bar(
    top_engines["unit_id"].astype(str),
    top_engines["mae"]
)

plt.xlabel("Engine ID")
plt.ylabel("MAE")

plt.title(
    "Highest-Error Engines"
)

plt.tight_layout()

plt.savefig(
    ENGINE_ERROR_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nFiles saved:")
print(OUTPUT_SUMMARY)
print(ERROR_PLOT)
print(RUL_ERROR_PLOT)
print(ENGINE_ERROR_PLOT)
print(ACTUAL_PREDICTED_PLOT)

print("\nError analysis completed successfully.")