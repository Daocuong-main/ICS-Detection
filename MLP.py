import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
import os
import joblib

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/model_mlp.pt"
SCALER_SAVE_PATH = "scalers/mlp_scaler.pkl"
RESULTS_DIR = "results/mlp"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# =======================
# 1. Load & Preprocess Data
# =======================
# Giả sử bạn đã lưu preprocessed_data_v4.pkl với các biến:
# X_train, X_val, X_test, y_train, y_val, y_test
X_train, X_val, X_test, y_train, y_val, y_test, *_= joblib.load('preprocessed_data1.pkl')

# Scale dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)

# =======================
# 2. Dataset & DataLoader
# =======================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'inputs': self.X[idx], 'labels': self.y[idx]}

BATCH_SIZE = 64
train_ds = TabularDataset(X_train_scaled, y_train)
val_ds   = TabularDataset(X_val_scaled, y_val)
test_ds  = TabularDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# =======================
# 3. Simple MLP Model
# =======================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =======================
# 4. Training Loop
# =======================
def train_mlp(model, train_loader, val_loader, device, epochs=30, lr=1e-3, patience=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    best_val_acc = 0
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device)
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
        print(f"[MLP] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        # Early stopping
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"⏹ [MLP] Early stop at epoch {epoch+1} (Val Acc did not improve)")
                break
    print(f"✅ [MLP] Best Val Acc: {best_val_acc:.4f}")

# =======================
# 5. Evaluation
# =======================
def evaluate_mlp(model, test_loader, device, results_dir=RESULTS_DIR):
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
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    print("F1-micro:", f1_micro)
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix - MLP')
    plt.savefig(f'{results_dir}/confusion_matrix.pdf')
    plt.savefig(f'{results_dir}/confusion_matrix.svg')
    plt.close()

# =======================
# 6. RUN TRAINING & EVAL
# =======================
input_dim = X_train_scaled.shape[1]
mlp_model = SimpleMLP(input_dim, num_classes=2)
train_mlp(mlp_model, train_loader, val_loader, DEVICE, epochs=30, lr=1e-3, patience=8)

mlp_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
evaluate_mlp(mlp_model, test_loader, DEVICE, results_dir=RESULTS_DIR)
