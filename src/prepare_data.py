from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train():
    path = RAW / "train_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download/extract NASA C-MAPSS into data/raw/."
        )
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df):
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    return df


if __name__ == "__main__":
    train = load_train()
    print("Shape:", train.shape)
    print("Engines:", train["unit_id"].nunique())
    print("Missing values:", int(train.isna().sum().sum()))

    train = add_rul(train)
    output = PROCESSED / "train_fd001_with_rul.csv"
    train.to_csv(output, index=False)
    print("Saved:", output)
