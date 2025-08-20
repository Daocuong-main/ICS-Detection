# src/training/train_DT.py
import os, json, time, joblib, numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier

# ---------- Config ----------
RUN_ID     = "v4"
DATA_PATH  = "data/processed/preprocessed_data_v4.pkl"  # adjust if needed
MODEL_DIR  = f"models/{RUN_ID}"
MODEL_KEY  = "dtree"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pkl")
SEED       = 42

# Decision Tree hyperparams (edit as you like)
CRITERION          = "gini"      # or "entropy", "log_loss"
MAX_DEPTH          = None
MIN_SAMPLES_SPLIT  = 2
MIN_SAMPLES_LEAF   = 1
RANDOM_STATE       = 42          # keep equal to SEED for reproducibility

os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s)

def main():
    set_seed(SEED)

    # 1) Load splits (expects: X_train, X_val, X_test, y_train, y_val, y_test, ...)
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

    # 2) Train on train + val (matches your original behavior)
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0).ravel()

    # 3) Model + fit with nanosecond timing
    dt = DecisionTreeClassifier(
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    t0_ns = time.perf_counter_ns()
    dt.fit(X_tr, y_tr)
    t1_ns = time.perf_counter_ns()

    # 4) Save model
    joblib.dump(dt, MODEL_PATH)

    # 5) Save training metadata for consistent later evaluation/plots
    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "sklearn_version": sklearn_version,
        "train_features": int(X_tr.shape[1]),
        "train_samples": int(X_tr.shape[0]),
        "class_distribution": {int(c): int((y_tr == c).sum()) for c in np.unique(y_tr)},
        "classes_order": [int(c) for c in dt.classes_],   # align proba columns later
        "dt_params": {
            "criterion": CRITERION,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        "train_time_ns_total": int(t1_ns - t0_ns),
        "seed": SEED,
        "device": "cpu",
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved Decision Tree model to {MODEL_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
