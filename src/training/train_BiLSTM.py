# src/training/train_BiLSTM.py
import os, json, time, numpy as np, joblib, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Config ----------
RUN_ID      = "v4"
DATA_PATH   = "data/processed/preprocessed_data_v4.pkl"  # adjust if needed
MODEL_DIR   = f"models/{RUN_ID}"
MODEL_KEY   = "bilstm"
MODEL_PATH  = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pt")

BATCH_SIZE  = 64
EPOCHS      = 30
PATIENCE    = 8
LR          = 1e-3
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ---------- Dataset ----------
class TabularSequenceDataset(Dataset):
    """
    Treat each feature vector as a length-T sequence with input_size=1:
      X[i] -> (seq_len=T, 1)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {"inputs": self.X[idx].unsqueeze(-1), "labels": self.y[idx]}

# ---------- Model ----------
class TabularBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, x):
        out, _ = self.bilstm(x)     # (B, T, 2*H)
        out = out[:, -1, :]         # last timestep
        return self.fc(out)

# ---------- Train ----------
def train(model, train_loader, val_loader, device, epochs=30, lr=1e-3, patience=8):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val_acc, best_state, bad = 0.0, None, 0

    t0_ns = time.perf_counter_ns()
    for ep in range(1, epochs+1):
        # Train
        model.train(); total = 0.0
        for b in train_loader:
            x, y = b["inputs"].to(device), b["labels"].to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        # Validate
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["inputs"].to(device), b["labels"].to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        val_acc = float((np.array(preds) == np.array(labs)).mean())
        print(f"[BiLSTM][{ep:02d}] train_loss={total/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, bad = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                print("Early stop.")
                break

    t1_ns = time.perf_counter_ns()
    return best_val_acc, best_state, int(t1_ns - t0_ns), ep

def main():
    set_seed(SEED)

    # 1) Load splits
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

    # 2) DataLoaders (no scaling required for sequence-as-1D)
    train_loader = DataLoader(TabularSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(TabularSequenceDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3) Model
    num_classes = int(len(np.unique(y_train)))
    model = TabularBiLSTM(input_size=1, hidden_size=64, num_layers=2, num_classes=num_classes, dropout=0.2).to(DEVICE)

    # 4) Train
    best_val_acc, best_state, train_time_ns, epochs_trained = train(
        model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR, patience=PATIENCE
    )

    # 5) Save best model + metadata
    torch.save(best_state, MODEL_PATH)
    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "architecture": "TabularBiLSTM",
        "params": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        "num_classes": int(num_classes),
        "seq_len": int(X_train.shape[1]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "epochs_trained": int(epochs_trained),
        "best_val_acc": float(best_val_acc),
        "train_time_ns_total": int(train_time_ns),
        "seed": SEED,
        "device": str(DEVICE),
        "paths": {"model": MODEL_PATH}
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved BiLSTM model to {MODEL_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
