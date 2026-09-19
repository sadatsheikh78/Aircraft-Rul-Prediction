from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "train_fd001_with_rul.csv"
OUTPUT = ROOT / "data" / "processed" / "train_fd001_features.csv"


def main():

    df = pd.read_csv(INPUT)

    print("=" * 60)
    print("TIME-SERIES FEATURE ENGINEERING")
    print("=" * 60)

    # Sort chronologically within each engine
    df = df.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Remove constant features
    # --------------------------------------------------

    candidate_features = [
        col
        for col in df.columns
        if col.startswith("sensor_")
        or col.startswith("setting_")
    ]

    constant_features = [
        col
        for col in candidate_features
        if df[col].nunique() <= 1
    ]

    print("\nRemoving constant features:")
    print(constant_features)

    df = df.drop(
        columns=constant_features
    )

    # --------------------------------------------------
    # Sensors for temporal feature engineering
    # --------------------------------------------------

    sensor_cols = [
        col
        for col in df.columns
        if col.startswith("sensor_")
    ]

    # --------------------------------------------------
    # Rolling features
    # --------------------------------------------------

    for sensor in sensor_cols:

        # Previous 5-cycle mean.
        # shift(1) ensures the current observation is
        # NOT included in its own historical mean.
        df[f"{sensor}_rolling_mean_5"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda x:
                x.shift(1).rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        # Previous 5-cycle standard deviation
        df[f"{sensor}_rolling_std_5"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda x:
                x.shift(1).rolling(
                    window=5,
                    min_periods=2
                ).std()
            )
        )

        # Change from previous cycle
        df[f"{sensor}_change"] = (
            df.groupby("unit_id")[sensor]
            .diff()
        )

    # --------------------------------------------------
    # Handle values created at the beginning of each
    # engine trajectory
    # --------------------------------------------------

    temporal_columns = [
        col
        for col in df.columns
        if (
            "rolling_std_5" in col
            or col.endswith("_change")
        )
    ]

    df[temporal_columns] = (
        df[temporal_columns]
        .fillna(0)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nOriginal columns :", 27)
    print("Final columns    :", df.shape[1])
    print("Rows             :", df.shape[0])

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()