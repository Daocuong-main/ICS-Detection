import joblib
import pandas as pd
import numpy as np

def load_and_preprocess_test(file_path, pipeline_path="X_full.pkl"):
    """
    Load and preprocess test data using the feature columns and scaler
    from the training pipeline (pipeline_path).
    Returns: (X_test_scaled, df_test, Y_test)
    """
    # Load scaler & feature columns from train pipeline
    data = joblib.load(pipeline_path)
    scaler = data["scaler"]
    columns = data["columns"]

    # Load and preprocess test file
    df_test = pd.read_csv(file_path)

    # Keep the label column for later if exists
    if 'Label' in df_test.columns:
        Y_test = df_test['Label'].values  # save before drop
    else:
        Y_test = None

    DROP_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label']
    df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns], inplace=True, errors='ignore')
    df_test = df_test[columns]  # keep only features used for training, correct order

    # Numeric cleaning
    df_test = df_test.apply(pd.to_numeric, errors='coerce')
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(df_test.median(numeric_only=True), inplace=True)
    df_test = df_test.astype('float32')

    # Scale with training scaler
    X_test_scaled = scaler.transform(df_test)

    print("Test data shape after scaling:", X_test_scaled.shape)
    return X_test_scaled, Y_test, df_test

# --- If run as script, test quick example ---
if __name__ == "__main__":
    X_test_scaled, Y_test, df_test = load_and_preprocess_test("datasets/mitm_hmi_scapy_forward_1plc_flow_labeled.csv")
