# src/training/train_boosting_RF.py
import os, json, time, numpy as np, joblib
from xgboost import XGBRFClassifier

# ---------- Config ----------
RUN_ID     = "v4"
DATA_PATH  = "data/processed/preprocessed_data_v4.pkl"  # adjust if needed
MODEL_DIR  = f"models/{RUN_ID}"
MODEL_KEY  = "xgbrf"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pkl")

SEED       = 42
os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s)

def main():
    set_seed(SEED)

    # 1) Load splits
    import joblib as _joblib
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = _joblib.load(DATA_PATH)

    # 2) Train on train+val (follows your previous pattern)
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)

    # 3) Define XGB Random Forest (no boosting)
    model = XGBRFClassifier(
        n_estimators=500,       # more trees, like RF
        max_depth=6,
        subsample=0.8,          # RF-style row subsampling
        colsample_bynode=0.8,   # RF-style feature subsampling per split
        reg_lambda=1.0,
        min_child_weight=1.0,
        eval_metric="logloss",
        tree_method="hist",     # set "gpu_hist" if you want GPU
        n_jobs=-1,
        random_state=SEED,
        verbosity=1
    )

    # 4) Train (timed)
    t0_ns = time.perf_counter_ns()
    model.fit(X_tr, y_tr)
    t1_ns = time.perf_counter_ns()

    # 5) Save model + metadata
    joblib.dump(model, MODEL_PATH)

    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "algorithm": "XGBRFClassifier",
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1
        },
        "train_shape": [int(X_tr.shape[0]), int(X_tr.shape[1])],
        "train_time_ns_total": int(t1_ns - t0_ns),
        "seed": SEED,
        "paths": {"model": MODEL_PATH},
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved XGBRF model to {MODEL_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
