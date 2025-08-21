# src/training/train_LSTM.py
import os, json, time, joblib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

from .utils import TabularSequenceDataset, set_seed

# ---------- Config ----------
RUN_ID      = "v4"
DATA_PATH   = "data/processed/preprocessed_data_v4.pkl"
MODEL_DIR   = f"models/{RUN_ID}"
MODEL_KEY   = "lstm"
MODEL_PATH  = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pt")

BATCH_SIZE  = 64
EPOCHS      = 30
PATIENCE    = 8
LR          = 1e-3
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM hyperparams
INPUT_SIZE  = 1         # we treat each feature as a time step, channel=1
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
NUM_CLASSES = 2

os.makedirs(MODEL_DIR, exist_ok=True)

class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: (B, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]             # last time step
        return self.fc(out)

def main():
    set_seed(SEED)

    # 1) Load data splits
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)
    seq_len = int(X_train.shape[1])

    # 2) Dataloaders (no scaling here to keep parity with your LSTM code)
    train_loader = DataLoader(TabularSequenceDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TabularSequenceDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3) Model + training loop (early stop by val acc), timed in ns
    model = TabularLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    best_val_acc, best_state, bad = 0.0, None, 0
    t0_ns = time.perf_counter_ns()
    for ep in range(1, EPOCHS + 1):
        # train
        model.train(); total = 0.0
        for b in train_loader:
            x, y = b["inputs"].to(DEVICE), b["labels"].to(DEVICE)
            logits = model(x)
            loss = crit(logits, y)
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
        print(f"[LSTM][{ep:02d}] train_loss={total/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, bad = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > PATIENCE:
                print("Early stop.")
                break
    t1_ns = time.perf_counter_ns()

    # 4) Save best model (.pt) + metadata JSON
    torch.save(best_state, MODEL_PATH)

    # classes order for consistency (e.g., aligning probabilities later)
    classes_sorted = [int(c) for c in sorted(np.unique(np.concatenate([y_train, y_val]).ravel()))]

    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "architecture": "TabularLSTM",
        "seq_len": seq_len,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "num_classes": NUM_CLASSES,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "classes_order": classes_sorted,
        "class_distribution_train": {int(c): int((np.array(y_train)==c).sum()) for c in np.unique(y_train)},
        "class_distribution_val":   {int(c): int((np.array(y_val)==c).sum())   for c in np.unique(y_val)},
        "epochs_trained": ep,
        "best_val_acc": best_val_acc,
        "train_time_ns_total": int(t1_ns - t0_ns),
        "seed": SEED,
        "device": str(DEVICE),
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved LSTM model to {MODEL_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
