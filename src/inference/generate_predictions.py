# Script to generate prediction files for all models without retraining
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..training.utils import TabularSequenceDataset

PRED_DIR = 'predictions'
os.makedirs(PRED_DIR, exist_ok=True)

# =====================
# Utility functions
# =====================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        X_arr = X.values if hasattr(X, 'values') else X
        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'inputs': self.X[idx], 'labels': self.y[idx]}

def save_npz(name, y_true, y_prob):
    np.savez(os.path.join(PRED_DIR, f'{name}.npz'), y_true=np.asarray(y_true), y_prob=np.asarray(y_prob))

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
# =====================
# Sklearn models
# =====================
def gen_decision_tree():
    path = os.path.join(PRED_DIR, 'DecisionTree.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    model = joblib.load('models/model_dtree.pkl')
    probs = model.predict_proba(X_test)[:, 1]
    save_npz('DecisionTree', y_test, probs)


def gen_random_forest():
    path = os.path.join(PRED_DIR, 'RandomForest.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    model = joblib.load('models/model_rf.pkl')
    probs = model.predict_proba(X_test)[:, 1]
    save_npz('RandomForest', y_test, probs)


def gen_xgboost():
    path = os.path.join(PRED_DIR, 'XGBoost.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    model = joblib.load('models/model_xgb.pkl')
    probs = model.predict_proba(X_test)[:, 1]
    save_npz('XGBoost', y_test, probs)

# =====================
# PyTorch models
# =====================
class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class TabularBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class FeatureTransformerV2(nn.Module):
    def __init__(self, n_feat, emb_dim=32, n_layers=6, n_heads=4, ff_factor=4, num_classes=2, dropout=0.1):
        super().__init__()
        d_model = emb_dim * 2
        self.val_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.col_embed = nn.Embedding(n_feat, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=d_model*ff_factor,
                                               dropout=dropout, activation='gelu',
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, N = x.size()
        v_emb = self.val_embed(x.unsqueeze(-1))
        c_emb = self.col_embed.weight[:N]
        tokens = torch.cat([v_emb, c_emb.expand(B, -1, -1)], dim=-1)
        tokens = self.drop(tokens)
        cls = self.cls_token.expand(B, 1, -1)
        tok_seq = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tok_seq)
        logits = self.head(encoded[:,0])
        return logits


def gen_lstm():
    path = os.path.join(PRED_DIR, 'LSTM.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    test_loader = DataLoader(TabularSequenceDataset(X_test, y_test), batch_size=64, shuffle=False)
    device = torch.device('cpu')
    model = TabularLSTM()
    model.load_state_dict(torch.load('models/model_lstm.pt', map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(device)
            out = model(x)
            probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
    save_npz('LSTM', y_test, probs)


def gen_bilstm():
    path = os.path.join(PRED_DIR, 'BiLSTM.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    test_loader = DataLoader(TabularSequenceDataset(X_test, y_test), batch_size=64, shuffle=False)
    device = torch.device('cpu')
    model = TabularBiLSTM()
    model.load_state_dict(torch.load('models/model_bilstm.pt', map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(device)
            out = model(x)
            probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
    save_npz('BiLSTM', y_test, probs)


def gen_tabtransformer():
    path = os.path.join(PRED_DIR, 'TabTransformer.npz')
    if os.path.exists(path):
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data_v4.pkl')
    scaler = joblib.load('scalers/tabtransformer_scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    test_loader = DataLoader(TabularDataset(X_test_scaled, y_test), batch_size=64, shuffle=False)
    device = torch.device('cpu')
    model = FeatureTransformerV2(n_feat=X_test_scaled.shape[1])
    model.load_state_dict(torch.load('models/model_tabtransformer.pt', map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(device)
            out = model(x)
            probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
    save_npz('TabTransformer', y_test, probs)
def gen_mlp():
    path = os.path.join(PRED_DIR, 'MLP.npz')
    if os.path.exists(path):
        return
    model_path = 'models/model_mlp.pt'
    scaler_path = 'scalers/mlp_scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print('MLP model or scaler not found, skipping.')
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data1.pkl')
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    test_loader = DataLoader(TabularDataset(X_test_scaled, y_test), batch_size=64, shuffle=False)
    device = torch.device('cpu')
    model = SimpleMLP(X_test_scaled.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(device)
            out = model(x)
            probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
    save_npz('MLP', y_test, probs)


def gen_secbert():
    path = os.path.join(PRED_DIR, 'SecBERT.npz')
    if os.path.exists(path):
        return
    model_path = 'models/model_secbert.pt'
    scaler_path = 'scalers/secbert_scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print('SecBERT model or scaler not found, skipping.')
        return
    try:
        from transformers import BertModel
    except Exception:
        print('Transformers library not available, skipping SecBERT.')
        return
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = joblib.load('preprocessed_data1.pkl')
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    SEQ_LEN = 1
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    y_arr = y_test.values if hasattr(y_test, 'values') else y_test
    X_seq, y_seq = create_sequences(X_test_scaled, y_arr, SEQ_LEN)
    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return {'inputs': self.X[idx], 'labels': self.y[idx]}
    test_loader = DataLoader(SeqDataset(X_seq, y_seq), batch_size=64, shuffle=False)
    class SecBERTClassifier(nn.Module):
        def __init__(self, input_dim, hidden_size=768, seq_len=1, num_classes=2):
            super().__init__()
            self.embedding = nn.Linear(input_dim, hidden_size)
            pe = self.sinusoidal_positional_embedding(seq_len + 1, hidden_size)
            self.register_buffer('pos_embedding', pe)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.bert = BertModel.from_pretrained('jackaduma/SecBERT')
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
            attn = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=x.device)
            outputs = self.bert(inputs_embeds=x, attention_mask=attn)
            cls_state = outputs.last_hidden_state[:, 0, :]
            return self.output_layer(cls_state)
    device = torch.device('cpu')
    model = SecBERTClassifier(input_dim=X_test_scaled.shape[1], seq_len=SEQ_LEN, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for b in test_loader:
            x = b['inputs'].to(device)
            logits = model(x)
            probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
    save_npz('SecBERT', y_seq, probs)
# =====================
# Bagged ensemble predictions (already saved)
# =====================
def gen_bagging():
    lstm_p = np.load(os.path.join('predictions_v3', 'bagged_lstm_probs.npy'))
    xgb_p = np.load(os.path.join('predictions_v3', 'bagged_xgb_probs.npy'))
    _, _, X_test, _, _, y_test, *_ = joblib.load('preprocessed_data_v3.pkl')
    y_test = 1 - y_test  # flip labels as in training script
    lstm_p = 1 - lstm_p
    xgb_p = 1 - xgb_p
    save_npz('Bagged_LSTM', y_test, lstm_p)
    save_npz('Bagged_XGB', y_test, xgb_p)
    save_npz('Combined_Ensemble', y_test, (lstm_p + xgb_p)/2)


if __name__ == '__main__':
    gen_decision_tree()
    gen_random_forest()
    gen_xgboost()
    gen_lstm()
    gen_bilstm()
    gen_tabtransformer()
    gen_mlp()
    gen_secbert()
    gen_bagging()
    print('Prediction files generated in', PRED_DIR)
