# ✈️ Aircraft Remaining Useful Life Prediction

An end-to-end machine learning and deep learning system for predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using multivariate sensor time-series data from the **NASA C-MAPSS FD001** dataset.

The project compares a tree-based regression baseline using **XGBoost** with a sequence-based **LSTM** model designed to capture temporal patterns in engine sensor measurements.

---

## 📌 Project Overview

Predictive maintenance aims to identify equipment degradation before failure occurs.

For aircraft engines, Remaining Useful Life represents the estimated number of operating cycles an engine can continue to operate before reaching the end of its useful life.

This project develops a complete RUL prediction pipeline:

```text
NASA C-MAPSS Sensor Data
          ↓
Data Validation
          ↓
RUL Construction
          ↓
Exploratory Data Analysis
          ↓
Constant Feature Removal
          ↓
Engine-Level Validation
          ↓
XGBoost Baseline
          ↓
Temporal Feature Experiment
          ↓
LSTM Sequence Model
          ↓
Official Test Evaluation
          ↓
Error Analysis
🎯 Objectives
Analyze multivariate aircraft engine sensor data.
Construct Remaining Useful Life targets.
Identify constant and informative sensor features.
Establish an XGBoost regression baseline.
Experiment with temporal sensor features.
Develop an LSTM model for sequential RUL prediction.
Evaluate models using MAE and RMSE.
Perform leakage-safe validation at the engine level.
Evaluate performance on the official C-MAPSS FD001 test set.
Analyze model errors across different RUL ranges.
📊 Dataset
NASA C-MAPSS FD001

The project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001 dataset.

Training Dataset
Property	Value
Engines	100
Observations	20,631
Input features	18
Missing values	0
Maximum RUL	361 cycles
Mean RUL	~107.81 cycles
Official Test Dataset
Property	Value
Engines	100
Observations	13,096
Input features	18
Missing values	0
Minimum RUL	7 cycles
Maximum RUL	340 cycles
Mean RUL	~141.24 cycles

The original dataset contains operating settings and sensor measurements collected over successive operating cycles.

🔎 Exploratory Data Analysis

The initial analysis identified 7 constant features:

setting_3
sensor_1
sensor_5
sensor_10
sensor_16
sensor_18
sensor_19

These features were removed because they contained no variation.

This reduced the model input space to:

18 useful model features

Sensor–RUL Relationships

Several sensors showed noticeable relationships with RUL during exploratory analysis.

Examples:

Feature	Correlation with RUL
sensor_11	-0.696
sensor_4	-0.679
sensor_15	-0.643
sensor_2	-0.606
sensor_17	-0.606
sensor_3	-0.585
sensor_8	-0.564
sensor_13	-0.563
sensor_20	0.629
sensor_21	0.636
sensor_7	0.657
sensor_12	0.671

These correlations were used for exploratory analysis and feature understanding, not as evidence of causality.

🧪 Validation Strategy

Because observations from the same engine form a time-dependent sequence, randomly splitting individual rows can introduce information leakage.

Instead, the internal validation process was performed at the engine level.

100 engines
     │
     ├── 80 engines → Training
     │
     └── 20 engines → Validation

This ensures that validation engines are not also present in the training data.

🌳 Model 1 — XGBoost Baseline

The first model established a tree-based regression baseline.

Configuration
XGBRegressor

n_estimators       = 500
max_depth          = 6
learning_rate      = 0.05
subsample          = 0.8
colsample_bytree   = 0.8
objective          = squarederror
random_state       = 42
Internal Validation
Metric	XGBoost
MAE	23.957
RMSE	31.542

The baseline provided a reference point for evaluating the sequence-based LSTM model.

⏱️ Temporal Feature Engineering Experiment

An additional experiment was performed to explicitly represent recent sensor history.

For varying sensors, the following features were generated:

Previous 5-cycle rolling mean
Previous 5-cycle rolling standard deviation
Cycle-to-cycle change

This expanded the feature set from 18 to 63 features.

Result
Model	MAE	RMSE
Baseline XGBoost	23.957	31.542
Temporal XGBoost	24.078	32.385

The temporal feature experiment did not improve validation performance in this configuration.

Rather than selecting a model only because it produced a more favorable result, the experiment was retained as part of the development process.

🧠 Model 2 — LSTM

The second model was designed specifically for multivariate time-series data.

The LSTM receives a sequence of 30 operating cycles containing 18 input features per cycle.

Architecture
30 cycles × 18 features
          ↓
      LSTM (64)
          ↓
       Dropout
          ↓
      LSTM (32)
          ↓
       Dropout
          ↓
       Dense (16)
          ↓
       RUL Output
Preprocessing

A StandardScaler was fitted using the training engines only.

The same fitted scaler was then applied to validation and test data.

This prevents test information from influencing the training preprocessing.

Training

The model used:

Adam optimizer
Mean Squared Error loss
MAE monitoring
Early stopping
Model checkpointing
Internal Validation
Metric	LSTM
MAE	18.384
RMSE	26.048
🏆 Official C-MAPSS FD001 Test Results

For the primary benchmark, predictions were evaluated at the final observed cycle of each test engine.

This produces:

100 test engines → 100 final-cycle predictions

Final Results
Model	MAE ↓	RMSE ↓
XGBoost	21.098	29.063
LSTM	16.385	23.101
LSTM Improvement

Compared with the XGBoost baseline:

MAE improvement  : 22.34%
RMSE improvement : 20.51%

The LSTM achieved lower MAE and RMSE than the XGBoost baseline on the official FD001 final-cycle evaluation.

📈 Extended Error Analysis

In addition to the primary final-cycle benchmark, the LSTM was evaluated across all usable test sequences.

This produced:

Test sequences : 10,196
Test engines   : 100
All-Cycle Diagnostic Performance
Metric	LSTM
MAE	28.399
RMSE	38.715
Median Absolute Error	21.822
Mean Error	-9.512
Maximum Absolute Error	160.665

The negative mean error indicates an overall tendency toward underprediction in this diagnostic evaluation.

🔬 Error by Remaining Useful Life

One of the key findings from the error analysis was that prediction accuracy varies substantially with the actual remaining lifetime of the engine.

Actual RUL	MAE
0–30	2.75
31–60	9.53
61–100	22.51
101–150	23.40
151–200	29.76
200+	83.95
Observation

The model is substantially more accurate when the engine is closer to the end-of-life region.

Prediction error increases for engines with very high remaining useful life.

This suggests that distinguishing subtle degradation patterns during the early lifecycle is more difficult than estimating RUL closer to failure.

🔍 Error Analysis

The project includes:

Prediction error distribution
Actual vs predicted RUL
Error by RUL range
Engine-level MAE
Highest-error engine analysis

These analyses are used to understand where the model performs well and where it struggles, rather than evaluating the model only through a single aggregate metric.

🔐 Leakage Prevention

Several measures were used to reduce data leakage:

Engine-Level Splitting

Training and validation engines were kept separate.

Training-Only Scaling

The LSTM scaler was fitted only on training data.

Separate Official Test Evaluation

The official test set was evaluated after model development.

Final-Cycle Benchmark

The primary test comparison uses the final observed cycle from each test engine.

🛠️ Technology Stack
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Deep Learning
TensorFlow
Keras
LSTM
Visualization
Matplotlib
Seaborn
Model Persistence
Joblib
📁 Project Structure
aircraft_rul_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│
├── src/
│   ├── prepare_data.py
│   ├── prepare_test_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_xgboost.py
│   ├── train_xgboost_features.py
│   ├── train_lstm.py
│   ├── evaluate_test_models.py
│   ├── final_cycle_evaluation.py
│   ├── error_analysis.py
│   └── model_comparison.py
│
├── README.md
├── requirements.txt
└── .gitignore
🚀 Running the Project

Clone the repository:

git clone <repository-url>
cd aircraft_rul_prediction

Install dependencies:

pip install -r requirements.txt

Prepare the training data:

python src/prepare_data.py

Run exploratory analysis:

python src/eda.py

Train the XGBoost baseline:

python src/train_xgboost.py

Generate temporal features:

python src/feature_engineering.py

Train the temporal XGBoost experiment:

python src/train_xgboost_features.py

Train the LSTM:

python src/train_lstm.py

Prepare official test data:

python src/prepare_test_data.py

Evaluate on the official test set:

python src/evaluate_test_models.py

Run final-cycle evaluation:

python src/final_cycle_evaluation.py

Run error analysis:

python src/error_analysis.py
⚠️ Limitations

This project is a research and portfolio prototype based on the NASA C-MAPSS FD001 simulation dataset.

It should not be interpreted as a certified aircraft safety system or a production aircraft monitoring system.

The project does not connect to live aircraft telemetry and does not provide real-time aircraft health monitoring.

Additionally, model performance can vary across different datasets, operating conditions, and engine populations.

🔮 Future Improvements

Potential extensions include:

Attention-based sequence models
Transformer architectures for time-series prediction
Remaining-life uncertainty estimation
More C-MAPSS subsets such as FD002, FD003 and FD004
Hyperparameter optimization
Model explainability using SHAP
Real-time sensor-stream simulation
Deployment as an interactive predictive-maintenance dashboard
📌 Key Takeaway

The project demonstrates an end-to-end approach to predictive maintenance using multivariate sensor time-series data.

The primary official FD001 evaluation showed:

XGBoost
MAE  = 21.098
RMSE = 29.063

LSTM
MAE  = 16.385
RMSE = 23.101

The LSTM reduced MAE by 22.34% and RMSE by 20.51% relative to the XGBoost baseline.

The error analysis also showed that prediction accuracy was strongest near the end-of-life region and decreased for engines with very high remaining useful life.

👨‍💻 Author

Sadat Sheikh

B.Tech — Artificial Intelligence & Data Science

Machine Learning · Data Science · Python · SQL

⭐ Project Focus

Predictive Maintenance | Time-Series Machine Learning | Deep Learning | Aircraft Engine RUL Prediction