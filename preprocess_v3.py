import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
# Load raw data
#benign = pd.read_csv('datasets/hmi_normal_6h_flow_label.csv')
scapy_modify = pd.read_csv('datasets/mitm_hmi_scapy_modify_1plc_6h_flow_label.csv')
fw_36plc = pd.read_csv('datasets/mitm_hmi_forward_36plc_6h_flow_label.csv')

# Drop unnecessary columns
cols_to_keeps = ['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Label']

# Keep only the specified columns
scapy_modify = scapy_modify[cols_to_keeps]
fw_36plc = fw_36plc[cols_to_keeps]

# Merge and encode labels
df = pd.concat([scapy_modify], ignore_index=True)
# Create and fit the encoder
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Create mapping dictionary
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print("Label to Integer Mapping:")
for label, code in label_mapping.items():
    print(f"'{label}' → {code}")

# Prepare features and target
X = df.drop(columns=['Label'])
y = df['Label']

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.mean(numeric_only=True))

# # ========= FEATURE SELECTION START =========
# # 1. Remove constant features
# vt = VarianceThreshold(threshold=0.0)
# X_vt = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

# # 2. Select K best features (change k as you need, ví dụ k=30)
# k = 50 if X_vt.shape[1] > 50 else X_vt.shape[1]  # không chọn quá nhiều nếu số feature ít
# selector = SelectKBest(score_func=f_classif, k=k)
# X_selected = selector.fit_transform(X_vt, y)

# # Lưu lại tên các cột còn lại
# selected_columns = X_vt.columns[selector.get_support()].tolist()
# # ========= FEATURE SELECTION END =========

# Split (lưu ý: phải split sau khi feature selection)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42, shuffle=True)

# Save, bạn cũng nên lưu lại thông tin selector để dùng lại khi predict/test
joblib.dump(
    (X_train, X_val, X_test, y_train, y_val, y_test),
    'preprocessed_data_v3.pkl'
)
print("Preprocessing done. Saved as 'preprocessed_data_v3.pkl'.")
print(f"Kept {len(cols_to_keeps)} features: {cols_to_keeps}")
