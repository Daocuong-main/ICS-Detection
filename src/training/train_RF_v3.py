"""Training and evaluation script for RandomForest on v3 dataset with label remapping."""

import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

DATA_PATH = "data/processed/preprocessed_data_v3.pkl"
MODEL_SAVE_PATH = "models/models_v3/model_rf.pkl"
RESULTS_DIR = "outputs/results_v3/randomforest"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.npz")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.json")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


def load_data():
    return joblib.load(DATA_PATH)


def train_model(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[RandomForestClassifier, int]:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    start = time.perf_counter_ns()
    model.fit(X_tr, y_tr)
    end = time.perf_counter_ns()
    joblib.dump(model, MODEL_SAVE_PATH)
    return model, end - start


def remap_labels(
    model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_test_fixed = 1 - y_test
    idx_attack = list(model.classes_).index(0)
    probs_fixed = model.predict_proba(X_test)[:, idx_attack]
    preds_fixed = (probs_fixed >= 0.5).astype(int)
    return y_test_fixed, preds_fixed, probs_fixed


def evaluate_model(
    model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, train_time_ns: int
) -> int:
    start = time.perf_counter_ns()
    y_test_fixed, preds_fixed, probs_fixed = remap_labels(model, X_test, y_test)
    end = time.perf_counter_ns()
    eval_time_ns = end - start

    np.savez(PREDICTIONS_PATH, y_test=y_test_fixed, preds=preds_fixed, probs=probs_fixed)

    report = classification_report(y_test_fixed, preds_fixed, output_dict=True)
    report["timing"] = {
        "train_time_ns": train_time_ns,
        "evaluate_time_ns": eval_time_ns,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("=== RandomForest v3 Classification Report ===")
    print(classification_report(y_test_fixed, preds_fixed))
    print("F1-micro:", f1_score(y_test_fixed, preds_fixed, average="micro"))
    print(f"Train time: {train_time_ns / 1e9:.4f}s  ({train_time_ns} ns)")
    print(f"Eval time:  {eval_time_ns / 1e9:.4f}s  ({eval_time_ns} ns)")

    cm = confusion_matrix(y_test_fixed, preds_fixed, normalize="true")
    ConfusionMatrixDisplay(cm).plot(values_format=".2%")
    plt.title("Confusion Matrix – RandomForest (labels fixed)")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
    plt.close()

    plot_curves(y_test_fixed, probs_fixed)
    return eval_time_ns


def plot_curves(y_true: np.ndarray, probs: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall, precision, label="RandomForest")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/roc_pr_curves.pdf")
    fig.savefig(f"{RESULTS_DIR}/roc_pr_curves.svg")
    plt.close(fig)


def main() -> None:
    need_train = not os.path.exists(MODEL_SAVE_PATH)
    need_eval = not os.path.exists(PREDICTIONS_PATH)

    model = None
    train_time_ns = 0

    if need_train or need_eval:
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = load_data()
        X_tr = np.concatenate([X_train, X_val], axis=0)
        y_tr = np.concatenate([y_train, y_val], axis=0)

    if need_train:
        model, train_time_ns = train_model(X_tr, y_tr)
    else:
        model = joblib.load(MODEL_SAVE_PATH)

    if need_eval:
        evaluate_model(model, X_test, y_test, train_time_ns)
    else:
        data = np.load(PREDICTIONS_PATH)
        y_test_fixed = data["y_test"]
        probs_fixed = data["probs"]
        plot_curves(y_test_fixed, probs_fixed)
        print("✅ Model and predictions found; skipped training and evaluation.")

    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Results & plots saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
