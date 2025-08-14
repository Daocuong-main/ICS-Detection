import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
from xgboost import XGBRFClassifier  # XGBoost's Random-Forest variant

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# ================================
# CONFIG
# ================================
DATA_PATH        = 'data/processed/preprocessed_data_v4.pkl'
MODEL_SAVE_PATH  = 'models/models_v4/model_xgbrf.pkl'
RESULTS_DIR      = 'outputs/results_v4/xgbrf'

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

# ================================
# LOAD DATA
# Assumes joblib contains:
# X_train, X_val, X_test, y_train, y_val, y_test, ...
# ================================
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

# Train on train + val (your previous pattern)
X_tr = np.concatenate([X_train, X_val], axis=0)
y_tr = np.concatenate([y_train, y_val], axis=0)

# ================================
# TRAIN: XGBoost Random Forest
# Notes:
# - XGBRFClassifier builds trees *without boosting* (RF-style).
# - Use subsample < 1 and colsample_bynode < 1 for RF behavior.
# - tree_method='hist' is fast on CPU; switch to 'gpu_hist' if you have GPU.
# ================================
xgbrf = XGBRFClassifier(
    n_estimators=500,        # more trees, like RF
    max_depth=6,
    subsample=0.8,           # RF-style row subsampling
    colsample_bynode=0.8,    # RF-style feature subsampling per split
    reg_lambda=1.0,
    min_child_weight=1.0,
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=SEED
)

# ===== TRAINING TIME =====
start_train = time.perf_counter_ns()
xgbrf.fit(X_tr, y_tr)
end_train = time.perf_counter_ns()

# Store both ns and seconds for clarity
train_time_ns = end_train - start_train

joblib.dump(xgbrf, MODEL_SAVE_PATH)

# ================================
# EVALUATE
# ================================
start_evaluate = time.perf_counter_ns()
preds = xgbrf.predict(X_test)
probs = xgbrf.predict_proba(X_test)[:, 1]  # probability of class 1 (binary)
end_evaluate = time.perf_counter_ns()

# Fix: evaluation duration should be end - start
evaluate_time_ns = end_evaluate - start_evaluate
time_per_sample = evaluate_time_ns / len(X_test)
# Classification report (+ timing info)
report = classification_report(y_test, preds, output_dict=True)

# Add timing metadata into the same JSON file
report["timing"] = {
    "train_time_ns": train_time_ns,
    "evaluate_time_ns": evaluate_time_ns
}

with open(f"{RESULTS_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("=== XGBRF Classification Report ===")
print(classification_report(y_test, preds))
print("F1-micro:", f1_score(y_test, preds, average='micro'))
print(f"Train time: {train_time_s:.4f}s  ({train_time_ns} ns)")
print(f"Eval time:  {evaluate_time_s:.4f}s  ({evaluate_time_ns} ns)")

# Confusion Matrix (normalized as percentage)
cm = confusion_matrix(y_test, preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='.2%')
plt.title("Confusion Matrix - XGBRF")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBRF')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/roc_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/roc_curve.svg")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probs)
plt.figure()
plt.plot(recall, precision, label='XGBRF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBRF')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/pr_curve.pdf")
plt.savefig(f"{RESULTS_DIR}/pr_curve.svg")
plt.close()

print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ Results & plots saved in: {RESULTS_DIR}")
