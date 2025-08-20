# src/training/train_SecBERT.py
import os, json, time, numpy as np, joblib, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== Config =====
RUN_ID        = "v4"
DATA_PATH     = "data/processed/preprocessed_data_v4.pkl"   # adjust if needed
MODEL_DIR     = f"models/{RUN_ID}"
SCALER_DIR    = f"scalers/{RUN_ID}"
MODEL_KEY     = "secbert"
MODEL_PATH    = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pt")
SCALER_PATH   = os.path.join(SCALER_DIR, f"{MODEL_KEY}_scaler.pkl")

BATCH_SIZE    = 64
EPOCHS        = 10
PATIENCE      = 5
LR            = 5e-5
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
HIDDEN_SIZE   = 768
NUM_CLASSES   = 2
SEQ_LEN       = 1      # change if you want longer temporal windows

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ---- Data utils ----
def create_sequences(X, y, seq_len):
    """Sliding window over rows: [i-seq_len, i) -> label at i."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)

class TimeseriesClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, seq_len, feat)
        self.y = torch.tensor(y,  dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

# ---- Model ----
class SecBERTClassifier(nn.Module):
    """
    Projects tabular tokens -> BERT hidden, prepends [CLS], adds sin-cos positions,
    runs through a BERT encoder (SecBERT or fallback), takes CLS for logits.
    """
    def __init__(self, input_dim, hidden_size, seq_len, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim   = input_dim
        self.seq_len     = seq_len

        self.embedding = nn.Linear(input_dim, hidden_size)

        pe = self._sinusoidal_pe(seq_len + 1, hidden_size)  # +1 for CLS
        self.register_buffer("pos_embedding", pe)            # (1, L+1, H)

        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Try SecBERT, fallback to bert-base-uncased if offline
        from transformers import BertModel
        try:
            self.bert = BertModel.from_pretrained("jackaduma/SecBERT")
        except Exception:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.head = nn.Linear(hidden_size, num_classes)

    def _sinusoidal_pe(self, L, H):
        position = torch.arange(0, L).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, H, 2) * (-np.log(10000.0) / H))
        pe = torch.zeros(L, H)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, H)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        B = x.size(0)
        x = self.embedding(x)                       # (B, L, H)
        cls = self.cls_token.expand(B, 1, -1)       # (B, 1, H)
        x = torch.cat([cls, x], dim=1)              # (B, L+1, H)
        x = x + self.pos_embedding[:, :x.size(1)]   # add PE

        attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=x.device)
        out = self.bert(inputs_embeds=x, attention_mask=attn_mask)
        cls_state = out.last_hidden_state[:, 0, :]  # (B, H)
        return self.head(cls_state)                 # (B, num_classes)

# ---- Train (early-stop on val accuracy) ----
def train(model, train_loader, val_loader, device, epochs=10, lr=5e-5, patience=5):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)

    best_val_acc, best_state, bad = 0.0, None, 0
    t0_ns = time.perf_counter_ns()

    for ep in range(1, epochs + 1):
        # train
        model.train(); total = 0.0
        for b in train_loader:
            x, y = b["inputs"].to(device), b["labels"].to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        # validate
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["inputs"].to(device), b["labels"].to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        val_acc = float((np.array(preds) == np.array(labs)).mean())
        print(f"[SecBERT][{ep:02d}] train_loss={total/len(train_loader):.4f}  val_acc={val_acc:.4f}")

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

    # 2) Scale (same as your original SecBERT code)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    # 3) Build sequences
    # Note: if your data is grouped by flow/session, build sequences per group instead.
    X_train_seq, y_train_seq = create_sequences(X_train_s, (y_train.values if hasattr(y_train, "values") else y_train), SEQ_LEN)
    X_val_seq,   y_val_seq   = create_sequences(X_val_s,   (y_val.values   if hasattr(y_val,   "values")   else y_val),   SEQ_LEN)

    # 4) Loaders
    train_loader = DataLoader(TimeseriesClassificationDataset(X_train_seq, y_train_seq),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TimeseriesClassificationDataset(X_val_seq,   y_val_seq),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) Model
    input_dim = X_train_seq.shape[-1]
    model = SecBERTClassifier(input_dim=input_dim, hidden_size=HIDDEN_SIZE,
                              seq_len=SEQ_LEN, num_classes=NUM_CLASSES)

    # 6) Train (early stop on val acc)
    best_val_acc, best_state, train_time_ns, epochs_trained = train(
        model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR, patience=PATIENCE
    )

    # 7) Save best model + metadata
    torch.save(best_state, MODEL_PATH)

    classes_sorted = [int(c) for c in sorted(np.unique(np.concatenate([y_train, y_val]).ravel()))]
    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "architecture": "SecBERTClassifier",
        "hidden_size": HIDDEN_SIZE,
        "seq_len": SEQ_LEN,
        "num_classes": NUM_CLASSES,
        "input_dim": int(input_dim),
        "train_samples": int(len(X_train_seq)),
        "val_samples": int(len(X_val_seq)),
        "classes_order": classes_sorted,
        "class_distribution_train": {int(c): int((np.array(y_train)==c).sum()) for c in np.unique(y_train)},
        "class_distribution_val":   {int(c): int((np.array(y_val)==c).sum())   for c in np.unique(y_val)},
        "epochs_trained": int(epochs_trained),
        "best_val_acc": float(best_val_acc),
        "train_time_ns_total": int(train_time_ns),
        "seed": SEED,
        "device": str(DEVICE),
        "paths": {
            "model": MODEL_PATH,
            "scaler": SCALER_PATH
        }
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved SecBERT model to {MODEL_PATH}")
    print(f"✅ Saved scaler to {SCALER_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
