import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

PREDICTIONS_DIR = 'predictions'
RESULTS_DIR = 'results_combined'
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of experiments and corresponding prediction files
experiments = [
    ('DecisionTree', 'DecisionTree.npz'),
    ('RandomForest', 'RandomForest.npz'),
    ('XGBoost', 'XGBoost.npz'),
    ('LSTM', 'LSTM.npz'),
    ('BiLSTM', 'BiLSTM.npz'),
    ('TabTransformer', 'TabTransformer.npz'),
    ('Bagged_LSTM', 'Bagged_LSTM.npz'),
    ('Bagged_XGB', 'Bagged_XGB.npz'),
    ('Combined_Ensemble', 'Combined_Ensemble.npz'),
]

# --------- Precision-Recall Curve Figure ---------
plt.figure(figsize=(6, 5))
for name, file in experiments:
    path = os.path.join(PREDICTIONS_DIR, file)
    if not os.path.exists(path):
        continue
    data = np.load(path)
    y_true = data['y_true']
    y_prob = data['y_prob']
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUC={pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.pdf'))
print(f'Precision-Recall curve saved to {os.path.join(RESULTS_DIR, "precision_recall_curve.pdf")}')

# --------- ROC Curve Figure ---------
plt.figure(figsize=(6, 5))
for name, file in experiments:
    path = os.path.join(PREDICTIONS_DIR, file)
    if not os.path.exists(path):
        continue
    data = np.load(path)
    y_true = data['y_true']
    y_prob = data['y_prob']
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.pdf'))
print(f'ROC curve saved to {os.path.join(RESULTS_DIR, "roc_curve.pdf")}')
