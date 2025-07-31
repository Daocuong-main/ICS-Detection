import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# Load raw data
#benign = pd.read_csv('datasets/hmi_normal_6h_flow_label.csv')
scapy_modify = pd.read_csv('datasets/mitm_hmi_scapy_modify_1plc_6h_flow_label.csv')
scapy_modify_12h = pd.read_csv('datasets/mitm_hmi_scapy_modify_1plc_12h_flow_label.csv')

# Step 2: Filter only the MITM rows from scapy_modify_12h
mitm_12h = scapy_modify_12h[scapy_modify_12h["Label"] == "MITM"].copy()

# Step 3: Rename "MITM" label to "mitm"
mitm_12h["Label"] = "mitm"

# Drop unnecessary columns
cols_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
#benign.drop(columns=cols_to_drop, errors='ignore', inplace=True)
scapy_modify.drop(columns=cols_to_drop, errors='ignore', inplace=True)
mitm_12h.drop(columns=cols_to_drop,errors='ignore', inplace=True)

# Merge and encode labels
df = pd.concat([scapy_modify, mitm_12h], ignore_index=True)
df['Label'] = df['Label'].replace({'normal': 0, 'mitm': 1})

# Prepare features and target
X = df.drop(columns=['Label'])
y = df['Label']

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.mean(numeric_only=True))

# ========= FEATURE SELECTION START =========
# 1. Remove constant features
vt = VarianceThreshold(threshold=0.0)
X_vt = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

# 2. Select K best features (change k as you need, ví dụ k=30)
k = 50 if X_vt.shape[1] > 50 else X_vt.shape[1]  # không chọn quá nhiều nếu số feature ít
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_vt, y)

# Lưu lại tên các cột còn lại
selected_columns = X_vt.columns[selector.get_support()].tolist()
# ========= FEATURE SELECTION END =========

# Split (lưu ý: phải split sau khi feature selection)
X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42, shuffle=True)

# Save, bạn cũng nên lưu lại thông tin selector để dùng lại khi predict/test
joblib.dump(
    (X_train, X_val, X_test, y_train, y_val, y_test, selected_columns, vt, selector),
    'preprocessed_data_v4.pkl'
)
print("Preprocessing done. Saved as 'preprocessed_data_v4.pkl'.")
print(f"Kept {len(selected_columns)} features: {selected_columns}")
