import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_TEST_PATH = "data/raw/test_FD001.txt"
RAW_RUL_PATH = "data/raw/RUL_FD001.txt"

OUTPUT_PATH = "data/processed/test_fd001_with_rul.csv"


# ============================================================
# COLUMN NAMES
# ============================================================

columns = (
    ["unit_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("PREPARING OFFICIAL C-MAPSS FD001 TEST DATA")
print("=" * 60)

test_df = pd.read_csv(
    RAW_TEST_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

rul_df = pd.read_csv(
    RAW_RUL_PATH,
    header=None,
    names=["official_rul"]
)


print(f"Test rows   : {len(test_df)}")
print(f"Test engines: {test_df['unit_id'].nunique()}")
print(f"RUL entries : {len(rul_df)}")


# ============================================================
# CALCULATE RUL FOR TEST DATA
# ============================================================

# Find the last observed cycle for every engine
max_cycles = (
    test_df
    .groupby("unit_id")["cycle"]
    .max()
    .reset_index()
    .rename(columns={"cycle": "max_test_cycle"})
)


# Attach the official RUL value for each engine
max_cycles["official_rul"] = rul_df["official_rul"].values


# Merge engine-level information back into test dataset
test_df = test_df.merge(
    max_cycles,
    on="unit_id",
    how="left"
)


# Calculate RUL for every observation
#
# RUL =
# last observed cycle
# + official RUL at the end of observation
# - current cycle

test_df["rul"] = (
    test_df["max_test_cycle"]
    + test_df["official_rul"]
    - test_df["cycle"]
)


# ============================================================
# VALIDATION
# ============================================================

print("\nDATASET OVERVIEW")
print("-" * 60)

print(f"Rows       : {len(test_df)}")
print(f"Columns    : {len(test_df.columns)}")
print(f"Engines    : {test_df['unit_id'].nunique()}")
print(f"Missing    : {test_df.isna().sum().sum()}")


print("\nRUL statistics:")
print(test_df["rul"].describe())


# ============================================================
# CHECK RUL VALUES
# ============================================================

print("\nRUL VALIDATION")
print("-" * 60)

print(
    f"Minimum RUL : {test_df['rul'].min():.2f}"
)

print(
    f"Maximum RUL : {test_df['rul'].max():.2f}"
)

print(
    f"Negative RUL values : "
    f"{(test_df['rul'] < 0).sum()}"
)


# ============================================================
# REMOVE HELPER COLUMNS
# ============================================================

test_df = test_df.drop(
    columns=["max_test_cycle", "official_rul"]
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

test_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nSaved:")
print(OUTPUT_PATH)

print("\nTest data preparation completed successfully.")