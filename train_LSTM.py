import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import json

# ====== CONFIG ======
DATA_PATH = 'preprocessed_data_v4.pkl'
MODEL_SAVE_PATH = 'models/model_lstm.pt'
RESULTS_DIR = 'results/lstm'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# ====== LOAD & PREPROCESS DATA ======
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

# Dataset & DataLoader cho LSTM (shape: [batch, seq_len, input_dim=1])
class TabularSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # Chuyển mỗi dòng thành (seq_len, 1)
        return {'inputs': self.X[idx].unsqueeze(-1), 'labels': self.y[idx]}

train_ds = TabularSequenceDataset(X_train, y_train)
val_ds   = TabularSequenceDataset(X_val, y_val)
test_ds  = TabularSequenceDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ====== LSTM MODEL ======
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # lấy hidden cuối cùng
        out = self.fc(out)
        return out

# ====== TRAINING LOOP ======
def train_lstm(model, train_loader, val_loader, device, epochs=30, lr=1e-3, patience=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    best_val_acc = 0
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device)  # (batch, seq_len, 1)
            labels = batch['labels'].to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
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
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        print(f"[LSTM] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"⏹ [LSTM] Early stop at epoch {epoch+1}")
                break
    print(f"✅ [LSTM] Best Val Acc: {best_val_acc:.4f}")

# ====== EVALUATE ======
def evaluate_lstm(model, test_loader, device, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    # Classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    with open(f'{results_dir}/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print(classification_report(all_labels, all_preds))
    print("F1-micro:", f1_score(all_labels, all_preds, average='micro'))
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix - LSTM')
    plt.savefig(f'{results_dir}/confusion_matrix.pdf')
    plt.savefig(f'{results_dir}/confusion_matrix.svg')
    plt.close()
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - LSTM')
    plt.legend()
    plt.savefig(f"{results_dir}/roc_curve.pdf")
    plt.savefig(f"{results_dir}/roc_curve.svg")
    plt.close()
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(recall, precision, label='LSTM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - LSTM')
    plt.legend()
    plt.savefig(f"{results_dir}/pr_curve.pdf")
    plt.savefig(f"{results_dir}/pr_curve.svg")
    plt.close()

# ====== RUN TRAINING & EVAL ======
input_dim = X_train.shape[1]
lstm_model = TabularLSTM(input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2)
train_lstm(lstm_model, train_loader, val_loader, DEVICE, epochs=30, lr=1e-3, patience=8)
lstm_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
evaluate_lstm(lstm_model, test_loader, DEVICE, results_dir=RESULTS_DIR)
