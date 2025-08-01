"""Utility for plotting confusion matrices from saved predictions."""

import os
from glob import glob
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use TrueType fonts in PDFs to avoid incompatibilities with some
# matplotlib/numpy combinations when embedding fonts.
plt.rcParams["pdf.fonttype"] = 42
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PREDICTIONS_DIR = "predictions"
RESULTS_DIR = "results_combined"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Automatically gather all prediction files so the script does not need
# to be updated when new experiments are added.
experiments = {}
for pred_file in sorted(glob(os.path.join(PREDICTIONS_DIR, "*.npz"))):
    name = os.path.splitext(os.path.basename(pred_file))[0]
    experiments.setdefault(name, pred_file)

if not experiments:
    raise SystemExit("No prediction files found")

# Also include any prediction CSVs that accompany prior results.
for csv_file in sorted(glob(os.path.join('results*', '*', '*_predictions.csv'))):
    name = os.path.splitext(os.path.basename(csv_file))[0].replace('_predictions','')
    experiments.setdefault(name, csv_file)

for name, path in experiments.items():
    if not os.path.exists(path):
        continue

    if path.endswith(".npz"):
        data = np.load(path)
        y_true = np.asarray(data["y_true"], dtype=int)
        y_prob = np.asarray(data["y_prob"], dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        counts = df["cm_label"].value_counts()
        tn = int(counts.get("TN", 0))
        fp = int(counts.get("FP", 0))
        fn = int(counts.get("FN", 0))
        tp = int(counts.get("TP", 0))
        cm = np.array([[tn, fp], [fn, tp]])
    else:
        continue
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(4, 4))
    disp.plot(colorbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    out_file_pdf = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.pdf")
    out_file_png = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(out_file_pdf)
    plt.savefig(out_file_png)
    plt.close()
    print(f"Saved {out_file_pdf} and {out_file_png}")