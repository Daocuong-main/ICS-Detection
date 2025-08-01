import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PREDICTIONS_DIR = 'predictions'
RESULTS_DIR = 'results_combined'
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of experiments and their prediction files
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

for name, file in experiments:
    path = os.path.join(PREDICTIONS_DIR, file)
    if not os.path.exists(path):
        continue

    data = np.load(path)
    y_true = data['y_true']
    y_prob = data['y_prob']
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {name}')
    out_file = os.path.join(RESULTS_DIR, f'{name}_confusion_matrix.pdf')
    plt.savefig(out_file)
    plt.close()
    print(f'Saved {out_file}')
