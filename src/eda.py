from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train_fd001_with_rul.csv"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# --------------------------------------------------
# 1. BASIC DATASET INFORMATION
# --------------------------------------------------

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows       : {df.shape[0]}")
print(f"Columns    : {df.shape[1]}")
print(f"Engines    : {df['unit_id'].nunique()}")
print(f"Missing    : {df.isna().sum().sum()}")

# --------------------------------------------------
# 2. IDENTIFY CONSTANT FEATURES
# --------------------------------------------------

feature_cols = [
    col for col in df.columns
    if col not in ["unit_id", "cycle", "rul"]
]

constant_features = [
    col for col in feature_cols
    if df[col].nunique() <= 1
]

print("\n" + "=" * 60)
print("CONSTANT FEATURES")
print("=" * 60)

print(constant_features)
print(f"Number of constant features: {len(constant_features)}")

# --------------------------------------------------
# 3. SENSOR CORRELATION WITH RUL
# --------------------------------------------------

sensor_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]

correlations = (
    df[sensor_cols + ["rul"]]
    .corr()["rul"]
    .drop("rul")
    .sort_values()
)

print("\n" + "=" * 60)
print("SENSOR CORRELATION WITH RUL")
print("=" * 60)

print(correlations)

# Save correlations
correlations.to_csv(
    RESULTS / "sensor_rul_correlations.csv",
    header=["correlation"]
)

# --------------------------------------------------
# 4. CORRELATION PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 7))

correlations.plot(kind="barh")

plt.title("Sensor Correlation with Remaining Useful Life")
plt.xlabel("Correlation with RUL")
plt.ylabel("Sensor")

plt.tight_layout()
plt.savefig(
    RESULTS / "sensor_rul_correlation.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 5. RUL DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["rul"],
    bins=40,
    kde=True
)

plt.title("Distribution of Remaining Useful Life")
plt.xlabel("RUL (cycles)")
plt.ylabel("Number of observations")

plt.tight_layout()
plt.savefig(
    RESULTS / "rul_distribution.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 6. SENSOR DEGRADATION EXAMPLES
# --------------------------------------------------

engine_id = 1

engine_data = df[df["unit_id"] == engine_id]

selected_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_11"
]

for sensor in selected_sensors:

    plt.figure(figsize=(9, 5))

    plt.plot(
        engine_data["cycle"],
        engine_data[sensor]
    )

    plt.xlabel("Operating Cycle")
    plt.ylabel(sensor)

    plt.title(
        f"{sensor} Trend - Engine {engine_id}"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS / f"{sensor}_engine_{engine_id}.png",
        dpi=150
    )

    plt.close()

print("\nEDA completed.")
print(f"Results saved to: {RESULTS}")