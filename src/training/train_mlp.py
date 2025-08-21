# src/training/train_mlp.py
import os, json, time, numpy as np, joblib, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Config ----------
RUN_ID     = "v4"
DATA_PATH  = "data/processed/preprocessed_data_v4.pkl"  # adjust if your file differs
MODEL_DIR  = f"models/{RUN_ID}"
SCALER_DIR = f"scalers/{RUN_ID}"
MODEL_KEY  = "MLP"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pt")
SCALER_PATH= os.path.join(SCALER_DIR, f"{MODEL_KEY}_scaler.pkl")
BATCH_SIZE = 64
EPOCHS     = 30
PATIENCE   = 8
LR         = 1e-3
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class TabularDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64, 32], num_classes=2, p=0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def main():
    set_seed(SEED)

    # 1) Load splits
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

    # 2) Scale (train/val) and save scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    # 3) DataLoaders
    train_loader = DataLoader(TabularDS(X_train_s, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TabularDS(X_val_s,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

    # 4) Model + training (early stop by val acc)
    model = SimpleMLP(input_dim=X_train_s.shape[1], num_classes=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    best_val_acc, best_state, bad = 0.0, None, 0
    t0_ns = time.perf_counter_ns()
    for ep in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for b in train_loader:
            x, y = b["inputs"].to(DEVICE), b["labels"].to(DEVICE)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        # validate
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["inputs"].to(DEVICE), b["labels"].to(DEVICE)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        val_acc = float((np.array(preds) == np.array(labs)).mean())
        print(f"[MLP][{ep:02d}] train_loss={total/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, bad = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > PATIENCE:
                print("Early stop.")
                break
    t1_ns = time.perf_counter_ns()

    # 5) Save best model + metadata
    torch.save(best_state, MODEL_PATH)
    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "input_dim": int(X_train_s.shape[1]),
        "epochs_trained": ep,
        "best_val_acc": best_val_acc,
        "train_time_ns_total": int(t1_ns - t0_ns),
        "seed": SEED,
        "device": str(DEVICE)
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved model to {MODEL_PATH}")
    print(f"✅ Saved scaler to {SCALER_PATH}")

if __name__ == "__main__":
    main()
