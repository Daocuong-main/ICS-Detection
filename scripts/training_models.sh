#!/bin/bash

# echo "Starting MLP.py"
# python MLP.py
# echo "Finished MLP.py"
# echo "----------------------------------------"

# echo "Starting SecBert_Classificastion.py"
# python SecBert_Classificastion.py
# echo "Finished SecBert_Classificastion.py"
# echo "----------------------------------------"

echo "Starting train_bagging_LSTM_test.py"
python src/training/train_bagging_LSTM_test.py
echo "Finished train_bagging_LSTM_test.py"
echo "----------------------------------------"

echo "Starting train_bagging_LSTM_v3.py"
python src/training/train_bagging_LSTM_v3.py
echo "Finished train_bagging_LSTM_v3.py"
echo "----------------------------------------"

echo "Starting train_bagging_LSTM.py"
python src/training/train_bagging_LSTM.py
echo "Finished train_bagging_LSTM.py"
echo "----------------------------------------"

echo "Starting train_BiLSTM.py"
python src/training/train_BiLSTM.py
echo "Finished train_BiLSTM.py"
echo "----------------------------------------"

echo "Starting train_DT.py"
python src/training/train_DT.py
echo "Finished train_DT.py"
echo "----------------------------------------"

echo "Starting train_LSTM.py"
python src/training/train_LSTM.py
echo "Finished train_LSTM.py"
echo "----------------------------------------"

echo "Starting train_RF_v3.py"
python src/training/train_RF_v3.py
echo "Finished train_RF_v3.py"
echo "----------------------------------------"

echo "Starting train_RF.py"
python src/training/train_RF.py
echo "Finished train_RF.py"
echo "----------------------------------------"

echo "Starting train_xgboost_lstm_2_v3.py"
python src/training/train_xgboost_lstm_2_v3.py
echo "Finished train_xgboost_lstm_2_v3.py"
echo "----------------------------------------"

echo "Starting train_xgboost_lstm.py"
python src/training/train_xgboost_lstm.py
echo "Finished train_xgboost_lstm.py"
echo "----------------------------------------"

echo "Starting train_XGBoost.py"
python src/training/train_XGBoost.py
echo "Finished train_XGBoost.py"
echo "----------------------------------------"

# echo "Starting Tranformer.py"
# python Tranformer.py
# echo "Finished Tranformer.py"
# echo "----------------------------------------"