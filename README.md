✈️ Aircraft Remaining Useful Life Prediction

An end-to-end machine learning and deep learning pipeline for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using multivariate sensor time-series data from the NASA C-MAPSS FD001 dataset.

The project compares an XGBoost regression baseline with a 30-cycle LSTM sequence model and includes leakage-safe validation, official test evaluation, and detailed error analysis.

🎯 Project Highlights

Built a complete RUL prediction pipeline from raw C-MAPSS data to model evaluation.

Performed exploratory analysis and removed 7 constant features.

Used engine-level train/validation splitting to reduce temporal data leakage.

Established an XGBoost baseline.

Tested explicit temporal feature engineering.

Developed a 30-cycle LSTM for sequential sensor data.

Evaluated both models on the official FD001 test set.

Performed final-cycle and all-cycle error analysis.

Investigated performance across different RUL ranges.

📊 Dataset

NASA C-MAPSS FD001

The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001 dataset contains simulated turbofan engine degradation trajectories recorded over successive operating cycles.

Training Data

Property

Value

Engines

100

Observations

20,631

Model input features

18

Missing values

0

Maximum RUL

361 cycles

Mean RUL

~107.81 cycles

Official Test Data

Property

Value

Engines

100

Observations

13,096

Model input features

18

Missing values

0

Minimum RUL

7 cycles

Maximum RUL

340 cycles

Mean RUL

~141.24 cycles

The original dataset contains operating settings and sensor measurements collected across engine operating cycles.

🔬 Exploratory Data Analysis

Seven constant features were identified and removed:

setting_3
sensor_1
sensor_5
sensor_10
sensor_16
sensor_18
sensor_19

After removing constant features and excluding engine identifiers from the model input, 18 useful features remained.

Sensor–RUL Relationships

Several features showed noticeable correlations with RUL:

Feature

Correlation

sensor_11

-0.696

sensor_4

-0.679

sensor_15

-0.643

sensor_2

-0.606

sensor_17

-0.606

sensor_3

-0.585

sensor_8

-0.564

sensor_13

-0.563

sensor_20

0.629

sensor_21

0.636

sensor_7

0.657

sensor_12

0.671

These correlations were used for exploratory analysis and feature understanding, not as evidence of causality.

🧪 Validation Strategy

Because observations belonging to the same engine form a time-dependent sequence, randomly splitting individual rows can introduce information leakage.

Instead, the internal validation split was performed at the engine level:

100 Engines
     │
     ├── 80 Engines → Training
     │
     └── 20 Engines → Validation

This keeps validation engines separate from the training engines.

🌳 Model 1 — XGBoost

XGBoost was used as the tree-based regression baseline.

Configuration

n_estimators       = 500
max_depth          = 6
learning_rate      = 0.05
subsample          = 0.8
colsample_bytree   = 0.8
objective          = squarederror
random_state       = 42

Internal Validation

Metric

XGBoost

MAE

23.957

RMSE

31.542

This baseline provides a reference point for evaluating the sequence-based LSTM model.

⏱️ Temporal Feature Engineering Experiment

An additional XGBoost experiment was performed to explicitly represent recent sensor history.

For varying sensors, the following features were generated:

Previous 5-cycle rolling mean

Previous 5-cycle rolling standard deviation

Cycle-to-cycle change

This expanded the feature set from 18 to 63 features.

Results

Model

MAE

RMSE

Baseline XGBoost

23.957

31.542

Temporal XGBoost

24.078

32.385

The temporal feature experiment did not improve validation performance in this configuration. It was retained as part of the development process rather than omitted because it produced a less favorable result.

🧠 Model 2 — LSTM

The second model was designed for multivariate time-series data.

Each training sample contains a sequence of 30 operating cycles × 18 input features.

Architecture

30 cycles × 18 features
          │
      LSTM (64)
          │
       Dropout
          │
      LSTM (32)
          │
       Dropout
          │
      Dense (16)
          │
      RUL Output

Preprocessing

A StandardScaler was fitted using training engines only and then applied to validation and test data.

This prevents information from validation or test data from influencing the training preprocessing.

Training

The model used:

Adam optimizer

Mean Squared Error loss

MAE monitoring

Early stopping

Model checkpointing

Internal Validation

Metric

LSTM

MAE

18.384

RMSE

26.048

🏆 Official FD001 Test Evaluation

For the primary benchmark, predictions were evaluated at the final observed cycle of each test engine.

This gives:

100 test engines
      ↓
100 final-cycle predictions

Final Results

Model

MAE ↓

RMSE ↓

XGBoost

21.098

29.063

LSTM

16.385

23.101

Relative Improvement

Compared with the XGBoost baseline:

MAE improvement: 22.34%

RMSE improvement: 20.51%

These figures are based on the project's final-cycle evaluation of the official FD001 test set.

📈 Extended Error Analysis

In addition to the primary final-cycle benchmark, the LSTM was evaluated across all usable test sequences.

Test sequences : 10,196
Test engines   : 100

All-Cycle Diagnostic Performance

Metric

LSTM

MAE

28.399

RMSE

38.715

Median Absolute Error

21.822

Mean Error

-9.512

Maximum Absolute Error

160.665

The negative mean error indicates an overall tendency toward underprediction in this diagnostic evaluation.

🔎 Error by Actual RUL

Actual RUL

MAE

0–30

2.75

31–60

9.53

61–100

22.51

101–150

23.40

151–200

29.76

200+

83.95

Observation

The model is substantially more accurate when the engine is closer to the end-of-life region.

Prediction error increases for engines with very high remaining useful life, suggesting that subtle degradation patterns during earlier lifecycle stages are harder to distinguish from the available sensor information.

🔬 Error Analysis Included

The project includes analysis of:

Prediction error distribution

Actual vs. predicted RUL

Error by RUL range

Engine-level MAE

Highest-error engines

The purpose is to understand where the model performs well and where it struggles rather than relying on a single aggregate metric.

🔐 Leakage Prevention

Several measures were used to reduce data leakage:

Engine-Level Splitting

Training and validation engines were kept separate.

Training-Only Scaling

The LSTM scaler was fitted only on training data.

Separate Official Test Evaluation

The official test set was evaluated after model development.

Final-Cycle Benchmark

The primary model comparison uses the final observed cycle from each test engine.

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

TensorFlow / Keras

LSTM

Visualization

Matplotlib

Seaborn

Model Persistence

Joblib

📁 Project Structure

aircraft-rul-prediction/
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

Raw C-MAPSS data, trained model binaries, generated prediction files, and plots are excluded from version control through .gitignore.

🚀 Running the Project

1. Clone the repository

git clone https://github.com/sadatsheikh78/Aircraft-Rul-Prediction.git
cd Aircraft-Rul-Prediction

2. Install dependencies

pip install -r requirements.txt

3. Prepare training data

python src/prepare_data.py

4. Run exploratory analysis

python src/eda.py

5. Train the XGBoost baseline

python src/train_xgboost.py

6. Generate temporal features

python src/feature_engineering.py

7. Train the temporal XGBoost experiment

python src/train_xgboost_features.py

8. Train the LSTM

python src/train_lstm.py

9. Prepare official test data

python src/prepare_test_data.py

10. Evaluate models on the official test set

python src/evaluate_test_models.py

11. Run final-cycle evaluation

python src/final_cycle_evaluation.py

12. Run error analysis

python src/error_analysis.py

⚠️ Limitations

This project is a research and portfolio prototype based on the NASA C-MAPSS FD001 simulation dataset.

It should not be interpreted as a certified aircraft safety system or a production aircraft monitoring system.

The project:

Does not use live aircraft telemetry.

Does not provide real-time aircraft health monitoring.

Is evaluated on a simulated dataset.

May show different performance on other datasets, operating conditions, or engine populations.

🔮 Future Improvements

Potential extensions include:

Attention-based sequence models

Transformer architectures for time-series prediction

Remaining-life uncertainty estimation

Evaluation on FD002, FD003, and FD004

Hyperparameter optimization

SHAP-based model explainability

Real-time sensor-stream simulation

Interactive predictive-maintenance dashboard

📌 Key Takeaway

This project demonstrates an end-to-end predictive-maintenance workflow using multivariate aircraft engine sensor time-series data.

Primary Official FD001 Benchmark

Model

MAE

RMSE

XGBoost

21.098

29.063

LSTM

16.385

23.101

The final-cycle evaluation showed lower MAE and RMSE for the LSTM compared with the XGBoost baseline on this FD001 test setup.

The error analysis also showed that prediction accuracy was strongest near the end-of-life region and decreased for engines with very high remaining useful life.

👨‍💻 Author

Sadat Sheikh

B.Tech — Artificial Intelligence & Data Science

Machine Learning · Data Science · Python · SQL

Project Focus

Predictive Maintenance · Time-Series Machine Learning · Deep Learning · Aircraft Engine RUL Prediction

⭐ If you find this project useful, consider giving the repository a star.
