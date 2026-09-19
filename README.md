#✈️ Aircraft Remaining Useful Life Prediction

End-to-end machine learning and deep learning pipeline for predicting Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS FD001 dataset.

The project compares XGBoost with a 30-cycle LSTM model and includes leakage-safe validation, official test evaluation, and error analysis.

#🎯 Highlights

Multivariate aircraft sensor time-series analysis

Engine-level train/validation split

XGBoost regression baseline

Temporal feature engineering experiment

30-cycle LSTM sequence model

Official FD001 test evaluation

Final-cycle and all-cycle error analysis

#📊 Dataset

NASA C-MAPSS FD001

Dataset

Engines

Observations

Features

Training

100

20,631

18

Official Test

100

13,096

18

Training RUL ranges from 0 to 361 cycles; official test RUL ranges from 7 to 340 cycles. No missing values were found.

#🔎 EDA

Seven constant features were removed:

setting_3, sensor_1, sensor_5, sensor_10,
sensor_16, sensor_18, sensor_19

Some of the strongest sensor–RUL correlations were:

Feature

Correlation

sensor_11

-0.696

sensor_4

-0.679

sensor_15

-0.643

sensor_12

0.671

sensor_7

0.657

sensor_21

0.636

These correlations were used for exploratory analysis, not as evidence of causality.

#🧪 Validation Strategy

Random row-level splitting can leak information because multiple observations belong to the same engine.

Therefore, validation was performed at the engine level:

100 Engines
├── 80 → Training
└── 20 → Validation

For the LSTM, StandardScaler was fitted on training engines only.

#🌳 XGBoost Baseline

Configuration:

n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8

Internal Validation

Metric

XGBoost

MAE

23.957

RMSE

31.542

#⏱️ Temporal Feature Experiment

For varying sensors, the project generated:

Previous 5-cycle rolling mean

Previous 5-cycle rolling standard deviation

Cycle-to-cycle change

This increased the feature count from 18 to 63.

Model

MAE

RMSE

XGBoost

23.957

31.542

Temporal XGBoost

24.078

32.385

The temporal feature experiment did not improve validation performance in this configuration and was retained as part of the development process.

#🧠 LSTM Model

The LSTM uses sequences of 30 operating cycles × 18 features.

30 × 18
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
RUL

Training used Adam, MSE loss, MAE monitoring, early stopping, and model checkpointing.

Internal Validation

Metric

LSTM

MAE

18.384

RMSE

26.048

#🏆 Official FD001 Test Results

The primary benchmark uses the final observed cycle of each test engine, producing 100 predictions for 100 engines.

Model

MAE ↓

RMSE ↓

XGBoost

21.098

29.063

LSTM

16.385

23.101

Compared with XGBoost:

22.34% lower MAE

20.51% lower RMSE

#📈 Error Analysis

The LSTM was also evaluated across 10,196 usable test sequences.

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

MAE by Actual RUL

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

The model was more accurate near the end-of-life region, while error increased for engines with very high remaining useful life.

#🔐 Leakage Prevention

Engine-level train/validation split

Training-only scaler fitting

Separate official test evaluation

Final-cycle benchmark for primary comparison

#🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras · LSTM · Matplotlib · Seaborn · Joblib

##📁 Project Structure

Aircraft-Rul-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
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
├── README.md
├── requirements.txt
└── .gitignore

Raw data, trained models, generated CSVs, and plots are excluded from Git through .gitignore.

#🚀 Run

git clone https://github.com/sadatsheikh78/Aircraft-Rul-Prediction.git
cd Aircraft-Rul-Prediction
pip install -r requirements.txt

python src/prepare_data.py
python src/eda.py
python src/train_xgboost.py
python src/feature_engineering.py
python src/train_xgboost_features.py
python src/train_lstm.py
python src/prepare_test_data.py
python src/evaluate_test_models.py
python src/final_cycle_evaluation.py
python src/error_analysis.py

⚠️ Limitations

This is a research and portfolio prototype based on the NASA C-MAPSS FD001 simulation dataset.

It does not use live aircraft telemetry, provide real-time aircraft monitoring, or represent a certified aircraft safety system.

🔮 Future Improvements

Transformer/attention-based time-series models

Uncertainty estimation

Evaluation on FD002, FD003, and FD004

Hyperparameter optimization

SHAP explainability

Interactive predictive-maintenance dashboard

##👨‍💻 Author

Sadat Sheikh
B.Tech — Artificial Intelligence & Data Science

Focus: Machine Learning · Data Science · Python · SQL · Predictive Maintenance
