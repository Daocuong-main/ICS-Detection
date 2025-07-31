import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
from sklearn.utils import resample

# ====== CONFIG ======
DATA_PATH        = 'preprocessed_data_v7.pkl'
MODELS_DIR       = 'models_v7'
RESULTS_DIR      = 'results_v7'
os.makedirs(MODELS_DIR, exist_ok=True)
for sub in ['lstm_bag', 'xgb_bag', 'combined']:
    os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_LSTM      = 100   # number of LSTM bags
N_XGB       = 100    # number of XGB bags
EPOCHS      = 100
PATIENCE    = 5
LR          = 1e-3
SEED        = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====== LOAD DATA ======
X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load(DATA_PATH)

# ====== DATASET & DATALOADER ======
class TabularSequenceDataset(Dataset):
    def __init__(self, X, y):
        if hasattr(X, "values"):
            self.X      = torch.tensor(X.values, dtype=torch.float32)
            self.feature_names = X.columns.tolist()
        else:
            self.X      = torch.tensor(X, dtype=torch.float32)
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        self.X_raw = self.X.numpy()
        self.y     = torch.tensor(y.values if hasattr(y, "values") else y,
                                  dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'inputs': self.X[idx].unsqueeze(-1),
            'labels': self.y[idx],
            'raw_features': self.X_raw[idx]  # Include raw features
        }

full_train_ds = TabularSequenceDataset(X_train, y_train)
val_ds        = TabularSequenceDataset(X_val,   y_val)
test_ds       = TabularSequenceDataset(X_test,  y_test)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ====== MODEL CLASSES ======
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.fc(out)

# ====== TRAIN / EVAL HELPERS ======
def train_one_lstm(model, train_loader, val_loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    model.to(device).train()
    best_acc, no_imp = 0, 0
    for ep in range(1, EPOCHS+1):
        total_loss = 0
        for b in train_loader:
            x, y = b['inputs'].to(device), b['labels'].to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b['inputs'].to(device), b['labels'].to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        acc = np.mean(np.array(preds)==np.array(labs))
        model.train()
        if acc > best_acc:
            best_acc, no_imp = acc, 0
        else:
            no_imp += 1
            if no_imp > PATIENCE:
                break
    return best_acc

def evaluate_and_plot(y_true, y_pred, y_prob, out_dir, name, test_dataset):
    # Assign confusion matrix labels
    cm_labels = np.zeros_like(y_true, dtype=object)
    cm_labels[(y_true == 0) & (y_pred == 0)] = 'TN'  # True Negative
    cm_labels[(y_true == 1) & (y_pred == 1)] = 'TP'  # True Positive
    cm_labels[(y_true == 0) & (y_pred == 1)] = 'FP'  # False Positive
    cm_labels[(y_true == 1) & (y_pred == 0)] = 'FN'  # False Negative
    
    feature_names = test_dataset.feature_names

    # Create DataFrame with features and confusion matrix labels
    output_df = pd.DataFrame(test_dataset.X_raw, columns=feature_names)
    output_df['cm_label'] = cm_labels
    
    # Export to CSV
    output_df.to_csv(f"{out_dir}/{name}_predictions.csv", index=False)
    
    # Classification report
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(f"{out_dir}/{name}_report.json", "w") as f:
        json.dump(rpt, f, indent=4)
    print(f"\n{name} Report:\n", classification_report(y_true, y_pred))
    print(f"{name} F1-micro:", f1_score(y_true, y_pred, average='micro'))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{name} Confusion")
    plt.savefig(f"{out_dir}/{name}_cm.pdf")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_roc.pdf")
    plt.close()
    
    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=name)
    plt.title(f"{name} PR")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_pr.pdf")
    plt.close()

# ====== 1) BAGGED LSTM ======
lstm_models = []
bagged_lstm_probs = []
for i in range(N_LSTM):
    idxs = resample(range(len(full_train_ds)),
                    replace=True,
                    n_samples=len(full_train_ds),
                    random_state=SEED+i)
    ds_boot = Subset(full_train_ds, idxs)
    loader_boot = DataLoader(ds_boot, batch_size=BATCH_SIZE, shuffle=True)
    m = TabularLSTM()
    acc = train_one_lstm(m, loader_boot, val_loader, DEVICE)
    print(f" LSTM bag {i+1} best val-acc: {acc:.4f}")
    torch.save(m.state_dict(), f"{MODELS_DIR}/lstm_bag_{i}.pt")
    lstm_models.append(m)
    m.to(DEVICE).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(DEVICE)
            logits = m(x)
            probs.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
    bagged_lstm_probs.append(probs)

bagged_lstm_probs = np.mean(bagged_lstm_probs, axis=0)
bagged_lstm_preds = (bagged_lstm_probs >= 0.5).astype(int)
evaluate_and_plot(
    y_test, bagged_lstm_preds, bagged_lstm_probs,
    os.path.join(RESULTS_DIR,'lstm_bag'), "Bagged_LSTM", test_ds
)

# ====== 2) BAGGED XGBOOST ======
# convert to NumPy and then reshape
X_flat     = X_train.to_numpy().reshape(len(X_train), -1)
y_flat     = y_train.to_numpy()
X_val_flat = X_val.to_numpy().reshape(len(X_val), -1)
y_val_arr  = y_val.to_numpy()
X_test_flat= X_test.to_numpy().reshape(len(X_test), -1)

xgb_models = []
bagged_xgb_probs = []
for i in range(N_XGB):
    Xi, yi = resample(
        np.vstack([X_flat, X_val_flat]),
        np.hstack([y_flat, y_val_arr]),
        replace=True,
        n_samples=len(X_flat)+len(X_val_flat),
        random_state=SEED+i
    )
    model = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=SEED+i
    )
    model.fit(Xi, yi)
    joblib.dump(model, f"{MODELS_DIR}/xgb_bag_{i}.pkl")
    xgb_models.append(model)
    bagged_xgb_probs.append(model.predict_proba(X_test_flat)[:,1])

bagged_xgb_probs = np.mean(bagged_xgb_probs, axis=0)
bagged_xgb_preds = (bagged_xgb_probs >= 0.5).astype(int)
evaluate_and_plot(
    y_test, bagged_xgb_preds, bagged_xgb_probs,
    os.path.join(RESULTS_DIR,'xgb_bag'), "Bagged_XGB", test_ds
)

# ====== 3) COMBINED BAGGED ENSEMBLE ======
combined_probs = (bagged_lstm_probs + bagged_xgb_probs) / 2
combined_preds = (combined_probs >= 0.5).astype(int)
evaluate_and_plot(
    y_test, combined_preds, combined_probs,
    os.path.join(RESULTS_DIR,'combined'), "Bagged_Ensemble", test_ds
)