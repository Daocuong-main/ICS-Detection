#!/bin/bash

# Train Bagging LSTM model
echo "Training Bagging LSTM model..."
python train_bagging_LSTM_v3.py

# Train XGBoost model
echo "Training XGBoost model..."
python train_xgboost_lstm_2_v3.py

# Train Random Forest model
echo "Training Random Forest model..."
python train_RF_v3.py