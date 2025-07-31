import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Directories
PREDICTIONS_DIR = 'predictions_v3'
RESULTS_DIR = 'results_v3'
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of experiments
experiments = [
    ('Bagged_LSTM', 'bagged_lstm_probs.npy'),
    ('Bagged_XGB', 'bagged_xgb_probs.npy'),
    ('Combined_Ensemble', 'combined_probs.npy')
]

# Load ground truth labels
y_test = np.load(os.path.join(PREDICTIONS_DIR, 'y_test.npy'))

# Initialize plots
plt.figure(figsize=(10, 5))

# PR Curve
plt.subplot(1, 2, 1)
for name, file in experiments:
    probs = np.load(os.path.join(PREDICTIONS_DIR, file))
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, label=f'{name} (AUC={auc(recall, precision):.2f})')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# ROC Curve
plt.subplot(1, 2, 2)
for name, file in experiments:
    probs = np.load(os.path.join(PREDICTIONS_DIR, file))
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Save combined plot
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'combined_curves.pdf'))
plt.show()

print("Combined PR and ROC curves have been saved to 'results_v3/combined_curves.pdf'.")
