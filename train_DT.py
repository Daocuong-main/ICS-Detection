import os
import joblib
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# ====== CONFIG ======
DATA_PATH = 'preprocessed_data_v4.pkl' 
MODEL_SAVE_PATH = 'models/model_dtree.pkl'
RESULTS_DIR = 'results/decisiontree'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== LOAD DATA ======
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)
# Có thể concat train + val
X_tr = np.concatenate([X_train, X_val], axis=0)
y_tr = np.concatenate([y_train, y_val], axis=0)

# ====== TRAIN Decision Tree ======
dtree = DecisionTreeClassifier(
    max_depth=None,       # Hoặc set max_depth=5,... nếu muốn regularize
    min_samples_split=2,
    random_state=42
)
dtree.fit(X_tr, y_tr)
joblib.dump(dtree, MODEL_SAVE_PATH)

# ====== EVALUATE Decision Tree ======
preds = dtree.predict(X_test)
probs = dtree.predict_proba(X_test)[:, 1] if len(dtree.classes_) > 1 else np.zeros_like(preds)

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
plt.title("Confusion Matrix - Decision Tree")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
plt.close()

# ROC Curve (chỉ vẽ nếu có xác suất class 1)
if len(dtree.classes_) > 1:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Decision Tree')
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/roc_curve.pdf")
    plt.savefig(f"{RESULTS_DIR}/roc_curve.svg")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision, label='Decision Tree')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Decision Tree')
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/pr_curve.pdf")
    plt.savefig(f"{RESULTS_DIR}/pr_curve.svg")
    plt.close()

print(f"✅ Kết quả và hình ảnh lưu tại: {RESULTS_DIR}")
