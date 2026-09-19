import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/train_fd001_with_rul.csv"

MODEL_PATH = "models/lstm_rul_model.keras"
SCALER_PATH = "models/lstm_scaler.joblib"

PREDICTION_PATH = "results/lstm_validation_predictions.csv"

LOSS_PLOT = "results/lstm_training_loss.png"
PREDICTION_PLOT = "results/lstm_predictions.png"

WINDOW_SIZE = 30
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LSTM RUL PREDICTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df = df.sort_values(
    ["unit_id", "cycle"]
).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Number of engines: {df['unit_id'].nunique()}")


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    col
    for col in df.columns
    if col not in ["unit_id", "cycle", "rul"]
    and df[col].nunique() <= 1
]

print("\nRemoving constant features:")
print(constant_features)

df = df.drop(
    columns=constant_features
)


# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col not in ["unit_id", "rul"]
]

print(
    f"\nNumber of input features: "
    f"{len(feature_columns)}"
)

print(
    f"Sequence window: "
    f"{WINDOW_SIZE} cycles"
)


# ============================================================
# ENGINE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(
        df,
        groups=df["unit_id"]
    )
)

train_engines = (
    df.iloc[train_idx]["unit_id"].unique()
)

val_engines = (
    df.iloc[val_idx]["unit_id"].unique()
)

train_df = df[
    df["unit_id"].isin(train_engines)
].copy()

val_df = df[
    df["unit_id"].isin(val_engines)
].copy()


print("\nENGINE SPLIT")
print("-" * 60)

print(
    f"Training engines   : "
    f"{len(train_engines)}"
)

print(
    f"Validation engines : "
    f"{len(val_engines)}"
)

print(
    f"Training rows      : "
    f"{len(train_df)}"
)

print(
    f"Validation rows    : "
    f"{len(val_df)}"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nFitting StandardScaler on training data...")

scaler = StandardScaler()

# IMPORTANT:
# Fit the scaler ONLY on training engines.

train_df[feature_columns] = (
    scaler.fit_transform(
        train_df[feature_columns]
    )
)


# Apply the exact same scaler
# to validation engines.

val_df[feature_columns] = (
    scaler.transform(
        val_df[feature_columns]
    )
)


# Save scaler for official test evaluation.

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"Scaler saved: {SCALER_PATH}"
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    data,
    features,
    window_size
):

    X = []
    y = []
    metadata = []

    for engine_id, engine_data in data.groupby(
        "unit_id"
    ):

        engine_data = engine_data.sort_values(
            "cycle"
        )

        feature_values = (
            engine_data[features].values
        )

        rul_values = (
            engine_data["rul"].values
        )

        cycles = (
            engine_data["cycle"].values
        )

        # Skip engines shorter than
        # the sequence window.

        if len(engine_data) < window_size:
            continue

        for i in range(
            window_size - 1,
            len(engine_data)
        ):

            sequence = feature_values[
                i - window_size + 1:
                i + 1
            ]

            target = rul_values[i]

            X.append(sequence)
            y.append(target)

            metadata.append({
                "unit_id": engine_id,
                "cycle": cycles[i]
            })

    return (
        np.array(
            X,
            dtype=np.float32
        ),

        np.array(
            y,
            dtype=np.float32
        ),

        pd.DataFrame(
            metadata
        )
    )


print("\nCreating training sequences...")

X_train, y_train, train_meta = (
    create_sequences(
        train_df,
        feature_columns,
        WINDOW_SIZE
    )
)


print("Creating validation sequences...")

X_val, y_val, val_meta = (
    create_sequences(
        val_df,
        feature_columns,
        WINDOW_SIZE
    )
)


print("\nSEQUENCE DATA")
print("-" * 60)

print(
    f"X_train shape : "
    f"{X_train.shape}"
)

print(
    f"y_train shape : "
    f"{y_train.shape}"
)

print(
    f"X_val shape   : "
    f"{X_val.shape}"
)

print(
    f"y_val shape   : "
    f"{y_val.shape}"
)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nMODEL ARCHITECTURE")
print("-" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining LSTM...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=40,
    batch_size=128,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print(
    "\nGenerating validation predictions..."
)

predictions = model.predict(
    X_val,
    verbose=0
).flatten()


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)


print("\n" + "=" * 60)
print("LSTM MODEL PERFORMANCE")
print("=" * 60)

print(
    f"MAE  : {mae:.3f}"
)

print(
    f"RMSE : {rmse:.3f}"
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_results = val_meta.copy()

prediction_results["actual_rul"] = (
    y_val
)

prediction_results["predicted_rul"] = (
    predictions
)

prediction_results["absolute_error"] = (
    np.abs(
        y_val - predictions
    )
)

prediction_results.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# TRAINING LOSS PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "LSTM Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    LOSS_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 6)
)

sample_size = min(
    1000,
    len(y_val)
)

plt.plot(
    y_val[:sample_size],
    label="Actual RUL"
)

plt.plot(
    predictions[:sample_size],
    label="Predicted RUL"
)

plt.xlabel(
    "Validation Sample"
)

plt.ylabel(
    "Remaining Useful Life"
)

plt.title(
    "LSTM: Actual vs Predicted RUL"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PREDICTION_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nFiles saved:")

print(
    f"Model       : "
    f"{MODEL_PATH}"
)

print(
    f"Scaler      : "
    f"{SCALER_PATH}"
)

print(
    f"Predictions : "
    f"{PREDICTION_PATH}"
)

print(
    f"Loss plot   : "
    f"{LOSS_PLOT}"
)

print(
    f"Prediction  : "
    f"{PREDICTION_PLOT}"
)

print(
    "\nLSTM training completed successfully."
)