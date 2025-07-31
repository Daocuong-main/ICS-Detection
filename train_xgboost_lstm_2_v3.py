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
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ====== CONFIG ======
DATA_PATH        = 'preprocessed_data_v3.pkl'
MODEL_LSTM_PATH  = 'models_v3/model_lstm.pt'
MODEL_XGB_PATH   = 'models_v3/model_xgb.pkl'
RESULTS_DIR      = 'results_v3'
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
               device, epochs=100, lr=1e-3, patience=8):
    # compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    model.to(device)
    best_f1       = 0.0
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
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['inputs'].to(device)
                y = batch['labels'].to(device)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                val_probs.extend(prob)
                val_loss += criterion(logits, y).item()
                val_labels.extend(y.cpu().numpy())

        val_probs  = np.array(val_probs)
        val_labels = np.array(val_labels)
        val_preds  = (val_probs >= 0.5).astype(int)

        val_f1 = f1_score(val_labels, val_preds, average='binary')
        print(f"[LSTM] Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            torch.save(model.state_dict(), MODEL_LSTM_PATH)
            best_f1 = val_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break
    print(f"✅ [LSTM] Best Val F1: {best_f1:.4f}")

def find_best_threshold(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"Optimal threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")
    return best_thresh

def evaluate_and_plot(y_true, y_pred, y_prob, out_dir, name):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"{out_dir}/{name}_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[{name}] Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"[{name}] F1-micro:", f1_score(y_true, y_pred, average='micro'))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{out_dir}/{name}_confusion_matrix.pdf")
    plt.close()

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
    seq_len = X_train.shape[1]
    lstm = TabularLSTM(input_size=1, hidden_size=64,
                       num_layers=2, num_classes=2,
                       dropout=0.2)
    train_lstm(lstm, train_loader, val_loader,
               DEVICE, epochs=100, lr=1e-3, patience=8)

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
            lstm_probs.extend(prob)
            lstm_labels.extend(y.cpu().numpy())
    lstm_probs  = np.array(lstm_probs)
    lstm_labels = np.array(lstm_labels)
    best_thresh = find_best_threshold(lstm_labels, lstm_probs)
    lstm_preds  = (lstm_probs >= best_thresh).astype(int)

    evaluate_and_plot(lstm_labels, lstm_preds, lstm_probs,
                      LSTM_RESULTS_DIR, "LSTM")

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat   = X_val.reshape(len(X_val), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb.fit(
        np.vstack([X_train_flat, X_val_flat]),
        np.hstack([y_train, y_val])
    )
    joblib.dump(xgb, MODEL_XGB_PATH)

    xgb_probs = xgb.predict_proba(X_test_flat)[:,1]
    best_thresh_xgb = find_best_threshold(y_test, xgb_probs)
    xgb_preds = (xgb_probs >= best_thresh_xgb).astype(int)

    evaluate_and_plot(y_test, xgb_preds, xgb_probs,
                      XGB_RESULTS_DIR, "XGBoost")

    ens_probs = (0.5 * lstm_probs + 0.5 * xgb_probs)
    best_thresh_ens = find_best_threshold(lstm_labels, ens_probs)
    ens_preds = (ens_probs >= best_thresh_ens).astype(int)

    evaluate_and_plot(lstm_labels, ens_preds, ens_probs,
                      ENS_RESULTS_DIR, "Ensemble")
