# src/training/train_bagging_LSTM.py
import os, json, time, numpy as np, joblib, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.utils import resample
from xgboost import XGBClassifier
from .utils import TabularSequenceDataset, set_seed

# ---------- Config ----------
RUN_ID      = "v4"
DATA_PATH   = f"data/processed/preprocessed_data_{RUN_ID}.pkl"  # adjust if needed

MODEL_DIR   = f"models/{RUN_ID}"
LSTM_DIR    = os.path.join(MODEL_DIR, "lstm_bag")
XGB_DIR     = os.path.join(MODEL_DIR, "xgb_bag")
MANIFEST    = os.path.join(MODEL_DIR, "bagging_manifest.json")

os.makedirs(LSTM_DIR, exist_ok=True)
os.makedirs(XGB_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
N_LSTM      = 20        # number of LSTM bags
N_XGB       = 20        # number of XGB bags
EPOCHS      = 20
PATIENCE    = 5
LR          = 1e-3
SEED        = 42
set_seed(SEED)

# ---------- Data ----------
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

full_train_ds = TabularSequenceDataset(X_train, y_train)
val_ds        = TabularSequenceDataset(X_val,   y_val)

val_loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Model ----------
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)       # (B, T, H)
        out = out[:, -1, :]         # last timestep
        return self.fc(out)

def train_one_lstm(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)

    best_acc, no_imp = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in train_loader:
            x, y = b["inputs"].to(device), b["labels"].to(device)
            logits = model(x)
            loss   = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # validate
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["inputs"].to(device), b["labels"].to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        acc = float((np.array(preds) == np.array(labs)).mean())

        if acc > best_acc:
            best_acc, no_imp = acc, 0
        else:
            no_imp += 1
            if no_imp > patience:
                break
    return best_acc

def main():
    manifest = {
        "run_id": RUN_ID,
        "seed": SEED,
        "device": str(DEVICE),
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "n_lstm": N_LSTM,
            "n_xgb": N_XGB
        },
        "outputs": {
            "lstm_bags": [],
            "xgb_bags": []
        }
    }

    # ===== 1) Bagged LSTM (bootstrap from train only; early-stop on val) =====
    print(f"==> Training {N_LSTM} LSTM bags …")
    for i in range(N_LSTM):
        # bootstrap indices from training set
        idxs = resample(range(len(full_train_ds)),
                        replace=True,
                        n_samples=len(full_train_ds),
                        random_state=SEED + i)
        ds_boot     = Subset(full_train_ds, idxs)
        loader_boot = DataLoader(ds_boot, batch_size=BATCH_SIZE, shuffle=True)

        m = TabularLSTM()
        t0 = time.perf_counter_ns()
        best_val_acc = train_one_lstm(m, loader_boot, val_loader, DEVICE)
        t1 = time.perf_counter_ns()

        # save
        path = os.path.join(LSTM_DIR, f"lstm_bag_{i}.pt")
        torch.save(m.state_dict(), path)

        manifest["outputs"]["lstm_bags"].append({
            "index": i,
            "path": path,
            "best_val_acc": best_val_acc,
            "train_time_ns": int(t1 - t0)
        })
        print(f"  [LSTM {i+1:02d}] best_val_acc={best_val_acc:.4f}  saved={path}")

    # ===== 2) Bagged XGBoost (bootstrap from train+val, flattened) =====
    print(f"==> Training {N_XGB} XGB bags …")
    X_train_np = np.asarray(X_train)
    X_val_np   = np.asarray(X_val)

    X_flat      = X_train_np.reshape(len(X_train_np), -1)
    X_val_flat  = X_val_np.reshape(len(X_val_np), -1)
    y_flat      = np.array(y_train)
    y_val_arr   = np.array(y_val)

    X_pool = np.vstack([X_flat, X_val_flat])
    y_pool = np.hstack([y_flat, y_val_arr])

    for i in range(N_XGB):
        Xi, yi = resample(
            X_pool, y_pool,
            replace=True,
            n_samples=len(X_pool),
            random_state=SEED + i
        )
        model = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED + i,
            n_jobs=-1,
            verbosity=1
        )
        t0 = time.perf_counter_ns()
        model.fit(Xi, yi)
        t1 = time.perf_counter_ns()

        path = os.path.join(XGB_DIR, f"xgb_bag_{i}.pkl")
        joblib.dump(model, path)

        manifest["outputs"]["xgb_bags"].append({
            "index": i,
            "path": path,
            "train_time_ns": int(t1 - t0),
            "params": {"n_estimators": 100, "eval_metric": "logloss", "random_state": SEED + i}
        })
        print(f"  [XGB {i+1:02d}] saved={path}")

    # ===== 3) Save manifest =====
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Bagging training complete.")
    print(f"📝 LSTM bags in: {LSTM_DIR}")
    print(f"📝 XGB bags in:  {XGB_DIR}")
    print(f"🧾 Manifest:     {MANIFEST}")

if __name__ == "__main__":
    main()
