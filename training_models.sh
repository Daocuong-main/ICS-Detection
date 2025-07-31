#!/bin/bash

echo "Running MLP.py"
python MLP.py

echo "Running SecBert_Classificastion.py"
python SecBert_Classificastion.py

echo "Running train_bagging_LSTM.py"
python train_bagging_LSTM.py

echo "Running train_bagging_LSTM_test.py"
python train_bagging_LSTM_test.py

echo "Running train_bagging_LSTM_v3.py"
python train_bagging_LSTM_v3.py

echo "Running train_BiLSTM.py"
python train_BiLSTM.py

echo "Running train_DT.py"
python train_DT.py

echo "Running train_LSTM.py"
python train_LSTM.py

echo "Running train_RF.py"
python train_RF.py

echo "Running train_RF_v3.py"
python train_RF_v3.py

echo "Running train_xgboost_lstm_2.py"
python train_xgboost_lstm_2.py

echo "Running train_XGBoost.py"
python train_XGBoost.py

echo "Running Tranformer.py"
python Tranformer.py

echo "Running tong_hop.py"
python tong_hop.py