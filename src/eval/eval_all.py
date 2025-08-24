# src/eval/eval_all.py
import os, glob, json, joblib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from xgboost import XGBClassifier
from tools.evalkit import save_run_artifacts, time_inference_ns

# -----------------------------
# Config (match your training)
# -----------------------------
RUN_ID        = "v4"
DATA_PATH     = f"data/processed/preprocessed_data_{RUN_ID}.pkl"
EVAL_DIR      = f"outputs/eval_{RUN_ID}"
MODELS_DIR    = f"models/{RUN_ID}"
SCALERS_DIR   = f"scalers/{RUN_ID}"
BATCH_SIZE    = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EVAL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)
y_test = np.array(y_test)

# Helpers: raw vs flattened views
X_test_raw  = X_test
X_test_flat = X_test.reshape(len(X_test), -1)

# -----------------------------
# Datasets
# -----------------------------
class SeqDS(Dataset):
    """For sequence models (LSTM/BiLSTM) where each row is a sequence."""
    def __init__(self, X, y):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"inputs": self.X[i].unsqueeze(-1), "labels": self.y[i]}

class TabularDS(Dataset):
    """For flat tabular models (Transformer)."""
    def __init__(self, X, y):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"inputs": self.X[i], "labels": self.y[i]}

# -----------------------------
# Model defs (match training)
# -----------------------------
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class TabularBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc     = nn.Linear(hidden_size*2, num_classes)
    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class FeatureTransformerV2(nn.Module):
    def __init__(self, n_feat, emb_dim=32, n_layers=6, n_heads=4, ff_factor=4, num_classes=2, dropout=0.1, label_smooth=0.05):
        super().__init__()
        d_model = emb_dim * 2
        self.label_smooth = label_smooth
        self.val_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.col_embed = nn.Embedding(n_feat, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ff_factor, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):        # x: (B, N_feat)
        B, N = x.size()
        v_emb = self.val_embed(x.unsqueeze(-1))         # (B,N,emb_dim)
        c_emb = self.col_embed.weight[:N]               # (N,emb_dim)
        tokens = torch.cat([v_emb, c_emb.expand(B, -1, -1)], dim=-1)  # (B,N,2*emb_dim)
        tokens = self.drop(tokens)
        cls = self.cls_token.expand(B, 1, -1)
        seq = torch.cat([cls, tokens], dim=1)           # (B,N+1,d_model)
        enc = self.encoder(seq)
        return self.head(enc[:, 0])
    def compute_loss(self, logits, targets):
        if self.label_smooth > 0:
            n_class = logits.size(1)
            smoothed = F.one_hot(targets, n_class).float()
            smoothed = smoothed * (1 - self.label_smooth) + self.label_smooth / n_class
            logp = F.log_softmax(logits, dim=1)
            return (-smoothed * logp).sum(dim=1).mean()
        return nn.CrossEntropyLoss()(logits, targets)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64, 32], num_classes=2, p=0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Eval helpers
# -----------------------------
@torch.no_grad()
def infer_probs_seq(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        x = b["inputs"].to(device)
        p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        out.append(p)
    return np.concatenate(out)

def load_scaler_or_identity(paths):
    """Try multiple scaler paths; if none found, return identity transform."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                pass
    class _Id:
        def transform(self, X): return X
    return _Id()

def print_done(model_key, out):
    print(f"\n[{model_key}] metrics:")
    print(json.dumps(out["metrics"], indent=2))

# -----------------------------
# 1) LSTM
# -----------------------------
def eval_lstm():
    path = os.path.join(MODELS_DIR, "LSTM.pt")
    if not os.path.exists(path):
        print("[lstm] skipped (model not found)."); return
    model = TabularLSTM().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    test_loader = DataLoader(SeqDS(X_test_raw, y_test), batch_size=BATCH_SIZE, shuffle=False)
    elapsed, probs = time_inference_ns(infer_probs_seq, model, test_loader, DEVICE)
    out = save_run_artifacts(EVAL_DIR, "lstm", y_test, probs, eval_time_ns_total=int(elapsed),
                             positive_class_name="attack", notes="LSTM on sequence rows")
    print_done("lstm", out)

# -----------------------------
# 2) BiLSTM
# -----------------------------
def eval_bilstm():
    path = os.path.join(MODELS_DIR, "bilstm.pt")
    if not os.path.exists(path):
        print("[bilstm] skipped (model not found)."); return
    model = TabularBiLSTM().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    test_loader = DataLoader(SeqDS(X_test_raw, y_test), batch_size=BATCH_SIZE, shuffle=False)
    elapsed, probs = time_inference_ns(infer_probs_seq, model, test_loader, DEVICE)
    out = save_run_artifacts(EVAL_DIR, "bilstm", y_test, probs, eval_time_ns_total=int(elapsed),
                             positive_class_name="attack", notes="BiLSTM on sequence rows")
    print_done("bilstm", out)

# -----------------------------
# 3) Tabular Transformer (uses scaler)
# -----------------------------
def eval_transformer():
    model_path  = os.path.join(MODELS_DIR, "tabtransformer.pt")
    scaler_paths = [
        os.path.join("scalers", RUN_ID, "tabtransformer_scaler.pkl"),
        os.path.join("scalers", "tabtransformer_scaler.pkl"),
    ]
    if not os.path.exists(model_path):
        print("[transformer] skipped (model not found)."); return
    scaler = load_scaler_or_identity(scaler_paths)
    X_test_scaled = scaler.transform(X_test_raw)
    model = FeatureTransformerV2(n_feat=X_test_scaled.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loader = DataLoader(TabularDS(X_test_scaled, y_test), batch_size=BATCH_SIZE, shuffle=False)
    @torch.no_grad()
    def _infer(loader, m, dev):
        m.eval(); out=[]
        for b in loader:
            x = b["inputs"].to(dev)
            out.append(torch.softmax(m(x), dim=1)[:,1].cpu().numpy())
        return np.concatenate(out)
    elapsed, probs = time_inference_ns(_infer, test_loader, model, DEVICE)
    out = save_run_artifacts(EVAL_DIR, "tabtransformer", y_test, probs, eval_time_ns_total=int(elapsed),
                             positive_class_name="attack", notes="Tabular Transformer (scaled)")
    print_done("tabtransformer", out)

# -----------------------------
# 4) SecBERT (uses scaler)
#     (We evaluate the linear head output the same way as training)
# -----------------------------
def eval_secbert():
    model_path  = os.path.join(MODELS_DIR, "secbert.pt")
    scaler_paths = [
        os.path.join("scalers", RUN_ID, "secbert_scaler.pkl"),
        os.path.join("scalers", "secbert_scaler.pkl"),
    ]
    if not os.path.exists(model_path):
        print("[secbert] skipped (model not found)."); return

    # --- must match train_SecBERT.py exactly ---
    from transformers import BertModel

    class SecBERTClassifier(nn.Module):
        def __init__(self, input_dim, hidden_size, seq_len, num_classes):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_dim   = input_dim
            self.seq_len     = seq_len

            self.embedding = nn.Linear(input_dim, hidden_size)

            # same sinusoidal positional embedding (buffer, not parameter)
            pe = self._sinusoidal_pe(seq_len + 1, hidden_size)  # +1 for CLS
            self.register_buffer("pos_embedding", pe)            # (1, L+1, H)

            self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            # Same backbone choice as training (SecBERT; fallback to bert-base)
            try:
                self.bert = BertModel.from_pretrained("jackaduma/SecBERT")
            except Exception:
                self.bert = BertModel.from_pretrained("bert-base-uncased")

            # IMPORTANT: name it 'head' to match checkpoint keys
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
            x = x + self.pos_embedding[:, :x.size(1)]   # add sinusoidal PE
            attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=x.device)
            out = self.bert(inputs_embeds=x, attention_mask=attn_mask)
            cls_state = out.last_hidden_state[:, 0, :]  # (B, H)
            return self.head(cls_state)                 # (B, num_classes)
    # --- end exact match ---

    # Load scaler and prepare test sequences (seq_len=1 like training)
    scaler = load_scaler_or_identity(scaler_paths)
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_seq = X_test_scaled[:, None, :]  # (N, 1, F)

    # Build model and load weights strictly (now names match)
    model = SecBERTClassifier(
        input_dim=X_test_scaled.shape[1],
        hidden_size=768,
        seq_len=1,
        num_classes=2
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)

    # Eval
    test_loader = DataLoader(
        TabularDS(X_test_seq, y_test),  # TabularDS will return (1, F) tensors → still fine; model expects (B, T, D)
        batch_size=BATCH_SIZE, shuffle=False
    )

    @torch.no_grad()
    def _infer(loader, m, dev):
        m.eval(); out=[]
        for b in loader:
            x = b["inputs"].to(dev)
            # ensure 3D (B, T, D); if TabularDS yields (B, 1, F) already, this is a no-op
            if x.ndim == 2:
                x = x.unsqueeze(1)
            out.append(torch.softmax(m(x), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    elapsed, probs = time_inference_ns(_infer, test_loader, model, DEVICE)
    out = save_run_artifacts(
        EVAL_DIR, "secbert", y_test, probs,
        eval_time_ns_total=int(elapsed),
        positive_class_name="attack",
        notes="SecBERT classifier (scaled, seq_len=1)"
    )
    print_done("secbert", out)


# -----------------------------
# 5) MLP (uses scaler)
# -----------------------------
def eval_mlp():
    model_path = os.path.join(MODELS_DIR, "MLP.pt")
    scaler_path = os.path.join(SCALERS_DIR, "MLP_scaler.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[mlp] skipped (model or scaler not found)."); return
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test_raw)
    model = SimpleMLP(input_dim=X_test_scaled.shape[1], num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loader = DataLoader(TabularDS(X_test_scaled, y_test), batch_size=BATCH_SIZE, shuffle=False)

    @torch.no_grad()
    def _infer(loader, m, dev):
        m.eval(); out = []
        for b in loader:
            x = b["inputs"].to(dev)
            out.append(torch.softmax(m(x), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    elapsed, probs = time_inference_ns(_infer, test_loader, model, DEVICE)
    out = save_run_artifacts(
        EVAL_DIR, "mlp", y_test, probs,
        eval_time_ns_total=int(elapsed),
        positive_class_name="attack",
        notes="Simple MLP (scaled tabular)"
    )
    print_done("mlp", out)


# -----------------------------
# 6) RandomForest / DecisionTree / XGB / XGBRF
# -----------------------------
def eval_sklearn_like(model_key, model_path, use_flat=False):
    if not os.path.exists(model_path):
        print(f"[{model_key}] skipped (model not found)."); return
    model = joblib.load(model_path) if model_path.endswith(".pkl") else None
    if model is None and model_path.endswith(".json"):
        # XGBoost saved as .json
        model = XGBClassifier()
        model.load_model(model_path)
    X_eval = X_test_flat if use_flat else X_test_raw
    def _pp(m, X): return m.predict_proba(X)[:, 1]
    elapsed, probs = time_inference_ns(_pp, model, X_eval)
    out = save_run_artifacts(EVAL_DIR, model_key, y_test, probs, eval_time_ns_total=int(elapsed),
                             positive_class_name="attack", notes=f"{model_key} eval")
    print_done(model_key, out)

def eval_rf():     eval_sklearn_like("Random Forest",     os.path.join(MODELS_DIR, "rf.pkl"),     use_flat=False)
def eval_dt():     eval_sklearn_like("Decision Tree",  os.path.join(MODELS_DIR, "dtree.pkl"),  use_flat=False)
def eval_xgb():    eval_sklearn_like("XGBoost",    os.path.join(MODELS_DIR, "XGBoost.json"),   use_flat=True)
def eval_xgbrf():  eval_sklearn_like("xgbrf",  os.path.join(MODELS_DIR, "xgbrf.pkl"),  use_flat=False)

# -----------------------------
# 7) Bagged LSTM / Bagged XGB
# -----------------------------
def eval_bagged_lstm():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "lstm_bag/lstm_bag_*.pt")))
    if not paths:
        print("[lstm_bag] skipped (no bag models found)."); return
    test_loader = DataLoader(SeqDS(X_test_raw, y_test), batch_size=BATCH_SIZE, shuffle=False)
    probs_bags = []
    total_elapsed = 0
    for p in paths:
        m = TabularLSTM().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        elapsed_sub, probs = time_inference_ns(infer_probs_seq, m, test_loader, DEVICE)
        probs_bags.append(probs)
        total_elapsed += elapsed_sub
    probs_mean = np.mean(np.stack(probs_bags, axis=0), axis=0)
    out = save_run_artifacts(EVAL_DIR, "lstm_bag", y_test, probs_mean, eval_time_ns_total=int(total_elapsed), notes=f"Mean of {len(paths)} LSTM bags")
    print_done("lstm_bag", out)

def eval_bagged_xgb():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "xgb_bag/xgb_bag_*.pkl")))
    if not paths:
        print("[xgb_bag] skipped (no bag models found)."); return
    probs_bags = []
    total_elapsed = 0
    def _pp(m, X): return m.predict_proba(X)[:, 1]
    for p in paths:
        m = joblib.load(p)
        elapsed_sub, probs = time_inference_ns(_pp, m, X_test_flat)
        probs_bags.append(probs)
        total_elapsed += elapsed_sub
    probs_mean = np.mean(np.stack(probs_bags, axis=0), axis=0)
    out = save_run_artifacts(EVAL_DIR, "xgb_bag", y_test, probs_mean, eval_time_ns_total=int(total_elapsed), notes=f"Mean of {len(paths)} XGB bags (flattened)")
    print_done("xgb_bag", out)

# -----------------------------
# 8) Simple Ensemble: (LSTM + XGB) / 2
# -----------------------------
def eval_ensemble_lstm_xgb():
    # Need both component models
    lstm_path = os.path.join(MODELS_DIR, "LSTM.pt")  # Updated to match the actual LSTM model file name
    xgb_path  = os.path.join(MODELS_DIR, "XGBoost.json")  # Updated to match the actual XGBoost model file name
    if not (os.path.exists(lstm_path) and os.path.exists(xgb_path)):
        print("[ensemble_avg] skipped (needs LSTM.pt and XGBoost.json)."); return

    # LSTM probs
    lstm = TabularLSTM().to(DEVICE)
    lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    test_loader = DataLoader(SeqDS(X_test_raw, y_test), batch_size=BATCH_SIZE, shuffle=False)
    elapsed_lstm, probs_lstm = time_inference_ns(infer_probs_seq, lstm, test_loader, DEVICE)

    # XGB probs
    if xgb_path.endswith(".pkl"):
        xgb = joblib.load(xgb_path)
    else:
        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
    def _pp(m, X): return m.predict_proba(X)[:, 1]
    elapsed_xgb, probs_xgb = time_inference_ns(_pp, xgb, X_test_flat)

    probs_ens = (probs_lstm + probs_xgb) / 2.0
    total_elapsed = elapsed_lstm + elapsed_xgb
    out = save_run_artifacts(EVAL_DIR, "ensemble_avg", y_test, probs_ens, eval_time_ns_total=int(total_elapsed), notes="Average of LSTM and XGB")
    print_done("ensemble_avg", out)

# -----------------------------
# Run everything that exists
# -----------------------------
if __name__ == "__main__":
    print(f"Evaluating on: {DATA_PATH}")
    eval_lstm()
    eval_bilstm()
    eval_transformer()
    eval_secbert()
    eval_mlp()
    eval_rf()
    eval_dt()
    eval_xgb()
    eval_xgbrf()
    eval_bagged_lstm()
    eval_bagged_xgb()
    eval_ensemble_lstm_xgb()
    print(f"\nAll available models evaluated. Artifacts in: {EVAL_DIR}")