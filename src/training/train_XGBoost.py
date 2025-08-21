# src/training/train_XGBoost.py
import os, json, time, numpy as np, joblib
from xgboost import XGBClassifier

# ---------- Config ----------
RUN_ID     = "v4"
DATA_PATH  = "data/processed/preprocessed_data_v4.pkl"  # adjust if needed
MODEL_DIR  = f"models/{RUN_ID}"
MODEL_KEY  = "XGBoost"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_KEY}.json")

SEED       = 42
os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s)

def main():
    set_seed(SEED)

    # 1) Load splits
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

    # 2) Train on train+val (common for tree models)
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)

    # 3) Define model (tweak as you wish)
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=1,
        # use_label_encoder is deprecated in recent xgboost; not needed here
    )

    # 4) Train (timed)
    t0_ns = time.perf_counter_ns()
    model.fit(X_tr, y_tr)
    t1_ns = time.perf_counter_ns()

    # 5) Save model + metadata
    model.save_model(MODEL_PATH)

    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "algorithm": "XGBClassifier",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": SEED,
            "n_jobs": -1,
        },
        "train_shape": [int(X_tr.shape[0]), int(X_tr.shape[1])],
        "train_time_ns_total": int(t1_ns - t0_ns),
        "seed": SEED,
        "paths": {"model": MODEL_PATH},
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved XGBoost model to {MODEL_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
