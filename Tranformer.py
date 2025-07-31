# =========================================
#        1. IMPORTS & CONFIG
# =========================================
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import json
import random

# ----- Fix seed for reproducibility -----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- CONFIG ---
INPUT_DIM = 50        # Sửa đúng số lượng feature bạn còn sau feature selection
NUM_CLASSES = 2
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/model_tabtransformer.pt"
SCALER_SAVE_PATH = "scalers/tabtransformer_scaler.pkl"
RESULTS_DIR = "results/tabtransformer"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# =========================================
#        2. LOAD & PREPROCESS DATA
# =========================================
# Bạn đã lưu đúng pipeline: nhớ load như sau:
X_train, X_val, X_test, y_train, y_val, y_test, *_= joblib.load('preprocessed_data_v4.pkl')

# --- Scale ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)

print("X_train_scaled shape:", X_train_scaled.shape)

# =========================================
#        3. Dataset & DataLoader
# =========================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'inputs': self.X[idx], 'labels': self.y[idx]}

train_ds = TabularDataset(X_train_scaled, y_train)
val_ds   = TabularDataset(X_val_scaled, y_val)
test_ds  = TabularDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# =========================================
#        4. TABULAR TRANSFORMER MODEL
# =========================================
class FeatureTransformerV2(nn.Module):
    def __init__(self,
                 n_feat: int,
                 emb_dim: int = 32,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 ff_factor: int = 4,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 label_smooth: float = 0.05):
        super().__init__()

        d_model = emb_dim * 2                        # ghép value-emb + col-emb
        self.label_smooth = label_smooth

        # 1) Value embedding (MLP 1→emb_dim)
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

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_factor,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        # 5) MLP classification head
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
        cls = self.cls_token.expand(B, -1, -1)           # (B,1,d_model)
        tok_seq = torch.cat([cls, tokens], dim=1)        # (B, N+1, d_model)
        encoded = self.encoder(tok_seq)                  # (B, N+1, d_model)
        logits  = self.head(encoded[:, 0])               # dùng CLS
        return logits

    def compute_loss(self, logits, targets):
        if self.label_smooth > 0:
            n_class = logits.size(1)
            smoothed_labels = F.one_hot(targets, n_class).float()
            smoothed_labels = smoothed_labels * (1 - self.label_smooth) \
                               + self.label_smooth / n_class
            log_prob = F.log_softmax(logits, dim=1)
            loss = (-smoothed_labels * log_prob).sum(dim=1).mean()
            return loss
        else:
            return nn.CrossEntropyLoss()(logits, targets)

# =========================================
#        5. TRAINING LOOP
# =========================================
def train_transformer(model, train_loader, val_loader, device, epochs=30, lr=5e-4, patience=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    best_val_acc = 0
    no_improve = 0
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            logits = model(inputs)
            loss = model.compute_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                labels = batch['labels'].to(device)
                logits = model(inputs)
                loss = model.compute_loss(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        # Early stopping
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"⏹ Early stop at epoch {epoch+1} (Val Acc did not improve)")
                break
    print(f"✅ Best Val Acc: {best_val_acc:.4f}")

# =========================================
#        6. TRAIN
# =========================================
model = FeatureTransformerV2(
    n_feat=X_train_scaled.shape[1],
    emb_dim=32,
    n_layers=6,
    n_heads=4,
    num_classes=2
)
train_transformer(model, train_loader, val_loader, DEVICE, epochs=30, lr=5e-4, patience=7)

# =========================================
#        7. EVALUATE & SAVE METRICS
# =========================================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['inputs'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(probs[:, 1])  # lấy xác suất class 1 (nếu binary)
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ---- Classification report ----
report = classification_report(all_labels, all_preds, output_dict=True)
with open(f'{RESULTS_DIR}/classification_report.json', 'w') as f:
    json.dump(report, f, indent=4)
print(classification_report(all_labels, all_preds))
f1_micro = f1_score(all_labels, all_preds, average='micro')
print("F1-micro:", f1_micro)

# ---- Confusion matrix ----
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix - Tabular Transformer')
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.pdf')
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.svg')
plt.close()

# ---- ROC Curve ----
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Tabular Transformer')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/roc_curve.pdf')
plt.savefig(f'{RESULTS_DIR}/roc_curve.svg')
plt.close()

# ---- Precision-Recall Curve ----
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
plt.figure()
plt.plot(recall, precision, label='Tabular Transformer')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Tabular Transformer')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/pr_curve.pdf')
plt.savefig(f'{RESULTS_DIR}/pr_curve.svg')
plt.close()

print(f"✅ Tabular Transformer pipeline complete! Report & plots saved to: {RESULTS_DIR}")
