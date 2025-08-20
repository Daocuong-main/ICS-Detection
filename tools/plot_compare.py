# tools/plot_compare.py
import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_metrics(run_dir: str) -> pd.DataFrame:
    rows = []
    for mdir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(mdir):
            continue
        name = os.path.basename(mdir)
        mj = os.path.join(mdir, "metrics.json")
        if not os.path.exists(mj):
            continue
        try:
            with open(mj) as f:
                metrics = json.load(f)
        except Exception:
            continue
        rows.append({
            "model": name,
            "accuracy": metrics.get("accuracy"),
            "f1_micro": metrics.get("f1_micro"),
            "auc_roc": metrics.get("auc_roc"),
            "auc_pr": metrics.get("auc_pr"),
            "n_samples": metrics.get("n_samples"),
            "eval_time_ns_total": metrics.get("eval_time_ns_total"),
            "eval_time_ns_per_sample": metrics.get("eval_time_ns_per_sample"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def plot_roc_all(run_dir: str, out_path: str, models: list | None = None):
    plt.figure()
    any_curve = False
    for mdir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(mdir):
            continue
        name = os.path.basename(mdir)
        if models and name not in models:
            continue

        p = os.path.join(mdir, "roc_curve.npz")
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p)
            fpr, tpr = d["fpr"], d["tpr"]
        except Exception:
            continue

        auc = None
        mj = os.path.join(mdir, "metrics.json")
        if os.path.exists(mj):
            try:
                with open(mj) as f:
                    auc = json.load(f).get("auc_roc", None)
            except Exception:
                pass

        label = f"{name} (AUC={auc:.3f})" if isinstance(auc, (int, float)) else name
        plt.plot(fpr, tpr, label=label)
        any_curve = True

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (cached)")
    if any_curve:
        plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_pr_all(run_dir: str, out_path: str, models: list | None = None):
    plt.figure()
    any_curve = False
    for mdir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(mdir):
            continue
        name = os.path.basename(mdir)
        if models and name not in models:
            continue

        p = os.path.join(mdir, "pr_curve.npz")
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p)
            precision, recall = d["precision"], d["recall"]
        except Exception:
            continue

        ap = None
        mj = os.path.join(mdir, "metrics.json")
        if os.path.exists(mj):
            try:
                with open(mj) as f:
                    ap = json.load(f).get("auc_pr", None)
            except Exception:
                pass

        label = f"{name} (AP={ap:.3f})" if isinstance(ap, (int, float)) else name
        plt.plot(recall, precision, label=label)
        any_curve = True

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (cached)")
    if any_curve:
        plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_confusions_all(run_dir: str, out_dir: str, models: list | None = None, normalize: str = "true"):
    """
    Loads cached y_test.npy and preds.npy for each model folder in run_dir
    and saves a confusion matrix for each model to out_dir as PDF and SVG.
    """
    os.makedirs(out_dir, exist_ok=True)
    any_cm = False

    for mdir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(mdir):
            continue
        name = os.path.basename(mdir)
        if models and name not in models:
            continue

        y_path = os.path.join(mdir, "y_test.npy")
        p_path = os.path.join(mdir, "preds.npy")
        if not (os.path.exists(y_path) and os.path.exists(p_path)):
            continue

        try:
            y_true = np.load(y_path)
            y_pred = np.load(p_path)
        except Exception:
            continue

        # Compute normalized confusion matrix (percent per true class)
        cm = confusion_matrix(y_true, y_pred, normalize=normalize if normalize else None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        plt.figure()
        fmt = ".2%" if normalize == "true" else ".2f" if normalize else "d"
        disp.plot(values_format=fmt, cmap=None, ax=plt.gca(), colorbar=False)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()

        pdf_path = os.path.join(out_dir, f"cm_{name}.pdf")
        svg_path = os.path.join(out_dir, f"cm_{name}.svg")
        plt.savefig(pdf_path)
        plt.savefig(svg_path)
        plt.close()

        any_cm = True

    if not any_cm:
        print(f"[plot_compare] No confusion matrices were generated (missing y_test.npy/preds.npy?).")

def export_table(run_dir: str, out_csv: str, sort_cols=("auc_pr","auc_roc","accuracy")):
    df = load_metrics(run_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if df.empty:
        df.to_csv(out_csv, index=False)
        print(f"[plot_compare] No metrics found in '{run_dir}'. Wrote empty table to: {out_csv}")
        return

    for c in sort_cols:
        if c not in df.columns:
            df[c] = np.nan

    df_sorted = df.sort_values(by=list(sort_cols), ascending=False)
    df_sorted.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Compare cached eval artifacts")
    ap.add_argument("--run_id", default="v4", help="Run id, e.g., v4")
    ap.add_argument("--eval_dir", default=None,
                    help="Directory with model subfolders (defaults to outputs/eval_<run_id>)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save plots/tables (defaults to outputs/compare_<run_id>)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional subset of model folder names to include")
    ap.add_argument("--cm_normalize", default="true", choices=["true","pred","all","none"],
                    help="Normalization for confusion matrices (sklearn normalize arg). 'none' = raw counts.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_dir = args.eval_dir or f"outputs/eval_{args.run_id}"
    out_dir = args.out_dir or f"outputs/compare_{args.run_id}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[plot_compare] Reading from: {run_dir}")
    print(f"[plot_compare] Writing to:   {out_dir}")

    # ROC & PR (PDF + SVG)
    plot_roc_all(run_dir, os.path.join(out_dir, "roc_all.pdf"), models=args.models)
    plot_roc_all(run_dir, os.path.join(out_dir, "roc_all.svg"), models=args.models)
    plot_pr_all(run_dir,  os.path.join(out_dir, "pr_all.pdf"), models=args.models)
    plot_pr_all(run_dir,  os.path.join(out_dir, "pr_all.svg"), models=args.models)

    # Confusion matrices (one per model) — saved as PDF & SVG
    plot_confusions_all(run_dir, os.path.join(out_dir, "confusions"), models=args.models,
                        normalize=None if args.cm_normalize == "none" else args.cm_normalize)

    # Table
    export_table(run_dir, os.path.join(out_dir, "metrics_table.csv"))
    print("✅ Combined plots & table saved.")
