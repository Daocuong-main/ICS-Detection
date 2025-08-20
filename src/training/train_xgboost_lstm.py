# src/training/train_xgboost_lstm.py
import os, json, time, joblib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier

# ---------- Config ----------
RUN_ID          = "v4"
DATA_PATH       = f"data/processed/preprocessed_data_{RUN_ID}.pkl"

MODEL_LSTM_PATH = f"models/models_{RUN_ID}/model_lstm.pt"
MODEL_XGB_PATH  = f"models/models_{RUN_ID}/model_xgb.pkl"
MANIFEST_PATH   = f"models/models_{RUN_ID}/xgb_lstm_training_manifest.json"

for p in [os.path.dirname(MODEL_LSTM_PATH), os.path.dirname(MODEL_XGB_PATH)]:
    os.makedirs(p, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 30
PATIENCE   = 8
LR         = 1e-3
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- Data ----------
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

class TabularSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {"inputs": self.X[idx].unsqueeze(-1), "labels": self.y[idx]}

train_loader = DataLoader(TabularSequenceDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TabularSequenceDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# ---------- LSTM ----------
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)       # (B, T, H)
        out = out[:, -1, :]         # last timestep
        return self.fc(out)

def train_lstm(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    best_val_acc, no_imp = 0.0, 0
    t0 = time.perf_counter_ns()
    for ep in range(1, epochs + 1):
        model.train()
        for b in train_loader:
            x, y = b["inputs"].to(device), b["labels"].to(device)
            loss = crit(model(x), y)
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
        val_acc = float((np.array(preds) == np.array(labs)).mean())
        print(f"[LSTM] epoch={ep:02d} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, no_imp = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp > patience:
                print("⏹ Early stop.")
                break
    t1 = time.perf_counter_ns()
    torch.save(best_state, MODEL_LSTM_PATH)
    return {"best_val_acc": best_val_acc, "epochs_trained": ep, "train_time_ns": int(t1 - t0)}

# ---------- XGBoost ----------
def train_xgb_on_flattened(X_train, X_val, y_train, y_val, seed=SEED):
    X_train_flat = np.asarray(X_train).reshape(len(X_train), -1)
    X_val_flat   = np.asarray(X_val).reshape(len(X_val), -1)
    X_tr = np.vstack([X_train_flat, X_val_flat])
    y_tr = np.hstack([np.array(y_train), np.array(y_val)])

    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=1
    )
    t0 = time.perf_counter_ns()
    xgb.fit(X_tr, y_tr)
    t1 = time.perf_counter_ns()
    joblib.dump(xgb, MODEL_XGB_PATH)
    return {"params": {"n_estimators": 100, "eval_metric": "logloss", "random_state": seed},
            "train_time_ns": int(t1 - t0)}

# ---------- Main ----------
if __name__ == "__main__":
    # 1) LSTM training (val used only for early stopping)
    lstm = TabularLSTM()
    lstm_info = train_lstm(lstm, train_loader, val_loader, DEVICE)

    # 2) XGBoost training on flattened (train+val)
    xgb_info = train_xgb_on_flattened(X_train, X_val, y_train, y_val)

    # 3) Manifest
    manifest = {
        "run_id": RUN_ID,
        "seed": SEED,
        "device": str(DEVICE),
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR
        },
        "outputs": {
            "lstm": {"path": MODEL_LSTM_PATH, **lstm_info},
            "xgb":  {"path": MODEL_XGB_PATH,  **xgb_info}
        }
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Training complete.")
    print(f"🧠 LSTM saved: {MODEL_LSTM_PATH}")
    print(f"🌳 XGBoost saved: {MODEL_XGB_PATH}")
    print(f"📄 Manifest: {MANIFEST_PATH}")
