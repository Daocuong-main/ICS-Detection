"""Training and evaluation script for Decision Tree classifier with caching."""

import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/processed/preprocessed_data_v4.pkl"
MODEL_SAVE_PATH = "models/models_v4/model_dtree.pkl"
RESULTS_DIR = "outputs/results_v4/decisiontree"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.npz")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.json")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


def load_data():
    return joblib.load(DATA_PATH)


def train_model(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[DecisionTreeClassifier, int]:
    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        random_state=SEED,
    )
    start = time.perf_counter_ns()
    model.fit(X_tr, y_tr)
    end = time.perf_counter_ns()
    joblib.dump(model, MODEL_SAVE_PATH)
    return model, end - start


def evaluate_model(
    model: DecisionTreeClassifier, X_test: np.ndarray, y_test: np.ndarray, train_time_ns: int
) -> int:
    start = time.perf_counter_ns()
    preds = model.predict(X_test)
    probs = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") and len(model.classes_) > 1
        else np.zeros_like(preds, dtype=float)
    )
    end = time.perf_counter_ns()
    eval_time_ns = end - start

    np.savez(PREDICTIONS_PATH, y_test=y_test, preds=preds, probs=probs)

    report = classification_report(y_test, preds, output_dict=True)
    report["timing"] = {
        "train_time_ns": train_time_ns,
        "evaluate_time_ns": eval_time_ns,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("=== Decision Tree Classification Report ===")
    print(classification_report(y_test, preds))
    print("F1-micro:", f1_score(y_test, preds, average="micro"))
    print(f"Train time: {train_time_ns / 1e9:.4f}s  ({train_time_ns} ns)")
    print(f"Eval time:  {eval_time_ns / 1e9:.4f}s  ({eval_time_ns} ns)")

    cm = confusion_matrix(y_test, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format=".2%")
    plt.title("Confusion Matrix - Decision Tree")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.svg")
    plt.close()

    if len(model.classes_) > 1:
        plot_curves(y_test, probs)
    return eval_time_ns


def plot_curves(y_true: np.ndarray, probs: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall, precision, label="Decision Tree")
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
        y_test = data["y_test"]
        probs = data["probs"]
        if len(np.unique(y_test)) > 1:
            plot_curves(y_test, probs)
        print("✅ Model and predictions found; skipped training and evaluation.")

    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Results & plots saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
