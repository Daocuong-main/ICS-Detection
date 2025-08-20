# tools/evalkit.py
import os, json, time, numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_ns() -> int:
    return time.perf_counter_ns()

def save_run_artifacts(
    run_dir: str,
    model_key: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    *,
    eval_time_ns_total: Optional[int] = None,
    positive_class_name: str = "attack",
    notes: str = ""
) -> Dict:
    """
    Saves all arrays + metrics + precomputed ROC/PR curves.
    No metrics are recomputed during replot; they are loaded from disk.
    """
    out_dir = os.path.join(run_dir, model_key)
    ensure_dir(out_dir)

    y_true = np.asarray(y_true).astype(int)
    probs  = np.asarray(probs).astype(float)
    if preds is None:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.asarray(preds).astype(int)

    # Metrics
    n = len(y_true)
    acc   = float(accuracy_score(y_true, preds))
    f1m   = float(f1_score(y_true, preds, average="micro"))
    try:
        auc_roc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr  = float(average_precision_score(y_true, probs))
    except ValueError:
        auc_pr  = float("nan")

    # Curves (precompute & cache to avoid recomputation on replot)
    fpr, tpr, roc_th = roc_curve(y_true, probs)
    prec, rec, pr_th = precision_recall_curve(y_true, probs)

    # Timing
    if eval_time_ns_total is None:
        eval_time_ns_total = 0
    eval_time_ns_per_sample = int(eval_time_ns_total // max(n, 1))

    # Save arrays
    np.save(os.path.join(out_dir, "y_test.npy"),  y_true)
    np.save(os.path.join(out_dir, "probs.npy"),   probs)
    np.save(os.path.join(out_dir, "preds.npy"),   preds)

    np.savez(os.path.join(out_dir, "roc_curve.npz"), fpr=fpr, tpr=tpr, thresholds=roc_th)
    np.savez(os.path.join(out_dir, "pr_curve.npz"),  precision=prec, recall=rec, thresholds=pr_th)

    # Save meta + metrics
    meta = {"model_key": model_key, "positive_class": positive_class_name, "notes": notes}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    metrics = {
        "accuracy": acc,
        "f1_micro": f1m,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "n_samples": n,
        "eval_time_ns_total": int(eval_time_ns_total),
        "eval_time_ns_per_sample": int(eval_time_ns_per_sample)
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"out_dir": out_dir, "metrics": metrics}

def time_inference_ns(fn, *args, **kwargs) -> Tuple[int, any]:
    """
    Measure total inference time (ns) for fn(*args, **kwargs).
    Return (elapsed_ns, result).
    """
    t0 = now_ns()
    res = fn(*args, **kwargs)
    t1 = now_ns()
    return t1 - t0, res
