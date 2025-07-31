import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt

# ====== CONFIG ======
DATA_PATH        = 'preprocessed_data_v4.pkl'
MODEL_LSTM_PATH  = 'models/model_lstm.pt'
MODEL_XGB_PATH   = 'models/model_xgb.pkl'
RESULTS_DIR      = 'results'
LSTM_RESULTS_DIR = os.path.join(RESULTS_DIR, 'lstm')
XGB_RESULTS_DIR  = os.path.join(RESULTS_DIR, 'xgb')
ENS_RESULTS_DIR  = os.path.join(RESULTS_DIR, 'ensemble')

for p in [os.path.dirname(MODEL_LSTM_PATH), os.path.dirname(MODEL_XGB_PATH),
          LSTM_RESULTS_DIR, XGB_RESULTS_DIR, ENS_RESULTS_DIR]:
    os.makedirs(p, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# ====== LOAD & SPLIT DATA ======
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

# ====== DATASET & DATALOADER ======
class TabularSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(
            y.values if hasattr(y, "values") else y,
            dtype=torch.long
        )
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {
            'inputs': self.X[idx].unsqueeze(-1),  # (seq_len, 1)
            'labels': self.y[idx]
        }

train_loader = DataLoader(TabularSequenceDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TabularSequenceDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TabularSequenceDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False)

# ====== LSTM MODEL ======
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        out    = out[:, -1, :]           # last time‐step
        return self.fc(out)

def train_lstm(model, train_loader, val_loader,
               device, epochs=30, lr=1e-3, patience=8):
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
    criterion     = nn.CrossEntropyLoss()
    model.to(device)
    best_val_acc  = 0.0
    no_improve    = 0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['inputs'].to(device)
            y = batch['labels'].to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # validate
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['inputs'].to(device)
                y = batch['labels'].to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y.cpu().numpy())
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_acc     = np.mean(np.array(val_preds)==np.array(val_labels))
        print(f"[LSTM] Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_LSTM_PATH)
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break
    print(f"✅ [LSTM] Best Val Acc: {best_val_acc:.4f}")

def evaluate_and_plot(y_true, y_pred, y_prob, out_dir, name):
    # classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"{out_dir}/{name}_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[{name}] Classification Report:")
    print(classification_report(y_true, y_pred))

    # F1
    print(f"[{name}] F1-micro:", f1_score(y_true, y_pred, average='micro'))

    # confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{out_dir}/{name}_confusion_matrix.pdf")
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_roc_curve.pdf")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_pr_curve.pdf")
    plt.close()

# ====== MAIN ======
if __name__ == "__main__":
    # 1) Train LSTM
    seq_len = X_train.shape[1]
    lstm = TabularLSTM(input_size=1, hidden_size=64,
                       num_layers=2, num_classes=2,
                       dropout=0.2)
    train_lstm(lstm, train_loader, val_loader,
               DEVICE, epochs=30, lr=1e-3, patience=8)

    # 2) Load best LSTM & get test preds
    lstm.load_state_dict(torch.load(MODEL_LSTM_PATH,
                                    map_location=DEVICE))
    lstm.to(DEVICE).eval()
    lstm_preds, lstm_probs, lstm_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['inputs'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            logits = lstm(x)
            prob   = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            pred   = (prob >= 0.5).astype(int)
            lstm_probs.extend(prob)
            lstm_preds.extend(pred)
            lstm_labels.extend(y.cpu().numpy())
    lstm_preds  = np.array(lstm_preds)
    lstm_probs  = np.array(lstm_probs)
    lstm_labels = np.array(lstm_labels)

    evaluate_and_plot(
        lstm_labels, lstm_preds, lstm_probs,
        LSTM_RESULTS_DIR, "LSTM"
    )

    # 3) Train XGBoost on flattened data
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat   = X_val.reshape(len(X_val), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(
        np.vstack([X_train_flat, X_val_flat]),
        np.hstack([y_train, y_val])
    )
    joblib.dump(xgb, MODEL_XGB_PATH)

    # 4) XGBoost test preds
    xgb_probs = xgb.predict_proba(X_test_flat)[:,1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    evaluate_and_plot(
        y_test, xgb_preds, xgb_probs,
        XGB_RESULTS_DIR, "XGBoost"
    )

    # 5) Ensemble by averaging probs
    ens_probs = (lstm_probs + xgb_probs) / 2
    ens_preds = (ens_probs >= 0.5).astype(int)

    evaluate_and_plot(
        lstm_labels, ens_preds, ens_probs,
        ENS_RESULTS_DIR, "Ensemble"
    )
