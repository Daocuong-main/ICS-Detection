# src/training/train_Transformer.py
import os, json, time, random, numpy as np, joblib, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============ Config ============
RUN_ID        = "v4"
DATA_PATH     = "data/processed/preprocessed_data_v4.pkl"   # adjust if needed
MODEL_DIR     = f"models/{RUN_ID}"
SCALER_DIR    = f"scalers/{RUN_ID}"
MODEL_KEY     = "tabtransformer"
MODEL_PATH    = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pt")
SCALER_PATH   = os.path.join(SCALER_DIR, f"{MODEL_KEY}_scaler.pkl")

BATCH_SIZE    = 64
EPOCHS        = 30
PATIENCE      = 7
LR            = 5e-4
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer hyperparams
EMB_DIM       = 32
N_LAYERS      = 6
N_HEADS       = 4
FF_FACTOR     = 4
DROPOUT       = 0.1
LABEL_SMOOTH  = 0.05

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ============ Data ============
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

# ============ Model ============
class FeatureTransformerV2(nn.Module):
    def __init__(self, n_feat, num_classes,
                 emb_dim=32, n_layers=6, n_heads=4, ff_factor=4,
                 dropout=0.1, label_smooth=0.05):
        super().__init__()
        d_model = emb_dim * 2
        self.label_smooth = label_smooth

        # 1) Value embedding
        self.val_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        # 2) Column embedding
        self.col_embed = nn.Embedding(n_feat, emb_dim)

        # 3) CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 4) Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*ff_factor,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        # 5) Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):                 # x: (B, N_feat)
        B, N = x.size()
        v_emb = self.val_embed(x.unsqueeze(-1))          # (B, N, emb_dim)
        c_emb = self.col_embed.weight[:N]                # (N, emb_dim)
        tokens = torch.cat([v_emb, c_emb.expand(B, -1, -1)], dim=-1)  # (B, N, 2*emb_dim)
        tokens = self.drop(tokens)
        cls = self.cls_token.expand(B, 1, -1)           # (B, 1, d_model)
        tok_seq = torch.cat([cls, tokens], dim=1)       # (B, N+1, d_model)
        enc = self.encoder(tok_seq)                     # (B, N+1, d_model)
        return self.head(enc[:, 0])                     # use CLS

    def compute_loss(self, logits, targets):
        if self.label_smooth and self.label_smooth > 0:
            n_class = logits.size(1)
            smoothed = F.one_hot(targets, n_class).float()
            smoothed = smoothed*(1-self.label_smooth) + self.label_smooth/n_class
            logp = F.log_softmax(logits, dim=1)
            return (-smoothed*logp).sum(dim=1).mean()
        return nn.CrossEntropyLoss()(logits, targets)

# ============ Train ============
def train(model, train_loader, val_loader, device, epochs=30, lr=5e-4, patience=7):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc, best_state, bad = 0.0, None, 0

    t0_ns = time.perf_counter_ns()
    for ep in range(1, epochs+1):
        # train
        model.train(); total = 0.0
        for b in train_loader:
            x, y = b["inputs"].to(device), b["labels"].to(device)
            logits = model(x)
            loss = model.compute_loss(logits, y)
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
        print(f"[TabTransformer][{ep:02d}] train_loss={total/len(train_loader):.4f}  val_acc={val_acc:.4f}")

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

    # 2) Scale train/val and save scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    # 3) Loaders
    train_loader = DataLoader(TabularDataset(X_train_s, y_train), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(TabularDataset(X_val_s,   y_val),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 4) Model
    num_classes = int(len(np.unique(y_train)))
    input_dim   = int(X_train_s.shape[1])
    model = FeatureTransformerV2(
        n_feat=input_dim, num_classes=num_classes,
        emb_dim=EMB_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        ff_factor=FF_FACTOR, dropout=DROPOUT, label_smooth=LABEL_SMOOTH
    ).to(DEVICE)

    # 5) Train
    best_val_acc, best_state, train_time_ns, epochs_trained = train(
        model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR, patience=PATIENCE
    )

    # 6) Save best model + metadata
    torch.save(best_state, MODEL_PATH)

    classes_sorted = [int(c) for c in sorted(np.unique(np.concatenate([y_train, y_val]).ravel()))]
    meta = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "architecture": "FeatureTransformerV2",
        "params": {
            "emb_dim": EMB_DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "ff_factor": FF_FACTOR, "dropout": DROPOUT, "label_smooth": LABEL_SMOOTH
        },
        "num_classes": int(num_classes),
        "input_dim": int(input_dim),
        "train_samples": int(len(X_train_s)),
        "val_samples": int(len(X_val_s)),
        "classes_order": classes_sorted,
        "class_distribution_train": {int(c): int((np.array(y_train)==c).sum()) for c in np.unique(y_train)},
        "class_distribution_val":   {int(c): int((np.array(y_val)==c).sum())   for c in np.unique(y_val)},
        "epochs_trained": int(epochs_trained),
        "best_val_acc": float(best_val_acc),
        "train_time_ns_total": int(train_time_ns),
        "seed": SEED,
        "device": str(DEVICE),
        "paths": {"model": MODEL_PATH, "scaler": SCALER_PATH}
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_KEY}_training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved TabTransformer model to {MODEL_PATH}")
    print(f"✅ Saved scaler to {SCALER_PATH}")
    print("📝 Training metadata saved next to the model.")

if __name__ == "__main__":
    main()
