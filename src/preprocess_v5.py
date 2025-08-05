import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Load raw data
# benign = pd.read_csv('datasets/hmi_normal_6h_flow_label.csv')
scapy_modify = pd.read_csv('datasets/mitm_hmi_scapy_modify_1plc_6h_flow_label.csv')
scapy_modify_12h = pd.read_csv('datasets/mitm_hmi_scapy_modify_1plc_12h_flow_label.csv')

# Filter only the MITM rows from scapy_modify_12h
mitm_12h = scapy_modify_12h[scapy_modify_12h["Label"] == "MITM"].copy()

# Rename "MITM" label to "mitm"
mitm_12h["Label"] = "mitm"

# Drop unnecessary columns
cols_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
# benign.drop(columns=cols_to_drop, errors='ignore', inplace=True)
scapy_modify.drop(columns=cols_to_drop, errors='ignore', inplace=True)
mitm_12h.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# Merge and encode labels
df = pd.concat([scapy_modify, mitm_12h], ignore_index=True)
df['Label'] = df['Label'].replace({'normal': 0, 'mitm': 1})

# Prepare features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Clean infinities and missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.mean(numeric_only=True))

# ====== NO FEATURE SELECTION – use all features ======

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
    shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
    shuffle=True
)

# Save preprocessed data
joblib.dump(
    (X_train, X_val, X_test, y_train, y_val, y_test),
    'preprocessed_data_full_features.pkl'
)

print("Preprocessing done. Saved as 'preprocessed_data_full_features.pkl'.")
print(f"Total features used: {X.shape[1]}")
