#!/bin/bash

# List of training scripts (without .py extension)
scripts=(
    "train_bagging_LSTM"
    "train_BiLSTM"
    "train_boosting_RF"
    "train_DT"
    "train_LSTM"
    "train_mlp"
    "train_RF"
    "train_SecBERT"
    "train_Tranformer"
    "train_xgboost_lstm"
    "train_XGBoost"
)

# Loop through each script and run it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python -m src.training.$script
    if [ $? -ne 0 ]; then
        echo "Error running $script. Stopping execution."
        exit 1
    fi
done

echo "All scripts executed successfully!"
