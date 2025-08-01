"""Utility for plotting confusion matrices from saved predictions."""

import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PREDICTIONS_DIR = "predictions"
RESULTS_DIR = "results_combined"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Automatically gather all prediction files so the script does not need
# to be updated when new experiments are added.
experiments = []
for pred_file in sorted(glob(os.path.join(PREDICTIONS_DIR, "*.npz"))):
    name = os.path.splitext(os.path.basename(pred_file))[0]
    experiments.append((name, pred_file))

if not experiments:
    raise SystemExit(f"No .npz prediction files found in {PREDICTIONS_DIR}")

for name, path in experiments:
    if not os.path.exists(path):
        continue

    data = np.load(path)
    y_true = np.asarray(data["y_true"], dtype=int)
    y_prob = np.asarray(data["y_prob"], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(4, 4))
    disp.plot(colorbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")