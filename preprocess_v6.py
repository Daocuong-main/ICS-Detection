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
cols_to_keeps = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
       'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
       'Flow IAT Std', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
       'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Std', 'Fwd Header Len',
       'Bwd Header Len', 'Fwd Pkts/s', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
       'Bwd Seg Size Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
       'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Bwd Win Byts',
       'Fwd Act Data Pkts', 'Active Mean', 'Active Max', 'Active Min',
       'Idle Mean', 'Idle Max', 'Idle Min', 'Label']

# Keep only the specified columns
scapy_modify = scapy_modify[cols_to_keeps]
mitm_12h = mitm_12h[cols_to_keeps]

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
    shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    shuffle=True
)

# Save preprocessed data
joblib.dump(
    (X_train, X_val, X_test, y_train, y_val, y_test),
    'preprocessed_data_v6.pkl'
)

print("Preprocessing done. Saved as 'preprocessed_data_v6.pkl'.")
print(f"Total features used: {X.shape[1]}")
