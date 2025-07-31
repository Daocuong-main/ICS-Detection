import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import json

# ====== CONFIG ======
DATA_PATH = 'preprocessed_data_v3.pkl'  # Đổi tên nếu bạn lưu file khác
MODEL_SAVE_PATH = 'models_v3/model_rf.pkl'
RESULTS_DIR = 'results_v3/randomforest'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== LOAD DATA ======
# Giả sử data đã split/scale thành:
# X_train, X_val, X_test, y_train, y_val, y_test
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

# Có thể concat X_train và X_val nếu muốn train trên all train data:
X_tr = np.concatenate([X_train, X_val], axis=0)
y_tr = np.concatenate([y_train, y_val], axis=0)

# ====== TRAIN RF ======
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)
joblib.dump(rf, MODEL_SAVE_PATH)

# ====== EVALUATE RF ======
preds = rf.predict(X_test)
probs = rf.predict_proba(X_test)[:, 1]  # Xác suất class 1 (binary)

# Classification report
report = classification_report(y_test, preds, output_dict=True)
with open(f"{RESULTS_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
print(classification_report(y_test, preds))
print("F1-micro:", f1_score(y_test, preds, average='micro'))

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
plt.close()

# ROC Curve (nếu bạn muốn)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

fpr, tpr, _ = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/roc_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/roc_curve.svg")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probs)
plt.figure()
plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/pr_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/pr_curve.svg")
plt.close()

print(f"✅ Kết quả và hình ảnh lưu tại: {RESULTS_DIR}")
