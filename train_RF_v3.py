import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
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
#
#  The model was trained with:       0 = MITM attack, 1 = normal
#  All your other experiments use:   0 = normal,     1 = MITM attack
#  We therefore flip labels **once** right here.

# ────────────────────────────────────────────────────────────────────────
# 🔄 REMAP LABELS FOR EVALUATION
# ────────────────────────────────────────────────────────────────────────
# 1) Ground truth
y_test_fixed = 1 - y_test            # 0↔1

# 2) Probabilities for the “attack” class
#    rf.predict_proba returns columns in the order rf.classes_
idx_attack   = list(rf.classes_).index(0)   # column that was ‘0’ during training
probs_fixed  = rf.predict_proba(X_test)[:, idx_attack]

# 3) Binary predictions after the remap
preds_fixed  = (probs_fixed >= 0.5).astype(int)
# ────────────────────────────────────────────────────────────────────────

# --- Standard reports/plots but using *_fixed ---
report = classification_report(y_test_fixed, preds_fixed, output_dict=True)
with open(f"{RESULTS_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
print(classification_report(y_test_fixed, preds_fixed))
print("F1‑micro:", f1_score(y_test_fixed, preds_fixed, average='micro'))

cm = confusion_matrix(y_test_fixed, preds_fixed)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix – Random Forest (labels fixed)")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
plt.close()



fpr, tpr, _   = roc_curve(y_test_fixed, probs_fixed)
auc_score     = roc_auc_score(y_test_fixed, probs_fixed)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Random Forest (labels fixed)')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/roc_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/roc_curve.svg")
plt.close()

precision, recall, _ = precision_recall_curve(y_test_fixed, probs_fixed)
plt.figure()
plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curve – Random Forest (labels fixed)')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/pr_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/pr_curve.svg")
plt.close()

print(f"✅ Kết quả đã hiệu chỉnh nhãn được lưu tại: {RESULTS_DIR}")


print(f"✅ Kết quả và hình ảnh lưu tại: {RESULTS_DIR}")
