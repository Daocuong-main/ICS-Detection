# =========================================
#        1. IMPORTS & CONFIG
# =========================================
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

# --- CONFIG ---
SEQ_LEN = 1         # sequence length (bạn chỉnh theo ý muốn)
INPUT_DIM = 30     # số lượng feature
NUM_CLASSES = 2      # binary classification (attack/benign)
BATCH_SIZE = 64
HIDDEN_SIZE = 768    # secbert-base
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/model_secbert.pt"
SCALER_SAVE_PATH = "scalers/secbert_scaler.pkl"
RESULTS_DIR = "results/secbert"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# =========================================
#        2. LOAD & PREPROCESS DATA
# =========================================
# Dữ liệu đã tiền xử lý: X_train, X_val, X_test, y_train, y_val, y_test
X_train, X_val, X_test, y_train, y_val, y_test, *_= joblib.load('preprocessed_data1.pkl')

# --- Scale ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)

# --- Sliding window tạo sequence ---
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, SEQ_LEN)
X_val_seq, y_val_seq     = create_sequences(X_val_scaled, y_val.values, SEQ_LEN)
X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test.values, SEQ_LEN)
print("X_train_scaled shape:", X_train_scaled.shape)

# =========================================
#        3. Dataset & DataLoader
# =========================================
class TimeseriesClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'inputs': self.X[idx], 'labels': self.y[idx]}

train_ds = TimeseriesClassificationDataset(X_train_seq, y_train_seq)
val_ds   = TimeseriesClassificationDataset(X_val_seq, y_val_seq)
test_ds  = TimeseriesClassificationDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# =========================================
#        4. SEC-BERT MODEL
# =========================================
class SecBERTClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        pe = self.sinusoidal_positional_embedding(seq_len + 1, hidden_size)
        self.register_buffer("pos_embedding", pe)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.bert = BertModel.from_pretrained("jackaduma/SecBERT")  # hoặc bert-base-uncased nếu không tải được SecBERT
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def sinusoidal_positional_embedding(self, seq_len, hidden_dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        cls_expanded = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_expanded, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1)]
        attention_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=x.device)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        cls_state = outputs.last_hidden_state[:, 0, :]
        logits = self.output_layer(cls_state)
        return logits

    def compute_loss(self, predictions, targets):
        return nn.CrossEntropyLoss()(predictions, targets)

# =========================================
#        5. TRAINING LOOP
# =========================================
def train_secbert(model, train_loader, val_loader, device, epochs=10, lr=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    best_val_acc = 0
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
        # Optionally save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
    print(f"✅ Best Val Acc: {best_val_acc:.4f}")


# =========================================
#        6. TRAIN
# =========================================
model = SecBERTClassifier(input_dim=X_train_scaled.shape[1], hidden_size=768, seq_len=SEQ_LEN, num_classes=2)

train_secbert(model, train_loader, val_loader, DEVICE, epochs=10, lr=5e-5)

# =========================================
#        7. EVALUATE & SAVE METRICS
# =========================================
# Load best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
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

# ---- Confusion matrix ----
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix - SecBERT')
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
plt.title('ROC Curve - SecBERT')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/roc_curve.pdf')
plt.savefig(f'{RESULTS_DIR}/roc_curve.svg')
plt.close()

# ---- Precision-Recall Curve ----
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
plt.figure()
plt.plot(recall, precision, label='SecBERT')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - SecBERT')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/pr_curve.pdf')
plt.savefig(f'{RESULTS_DIR}/pr_curve.svg')
plt.close()

print(f"✅ SecBERT pipeline complete! Report & plots saved to: {RESULTS_DIR}")

