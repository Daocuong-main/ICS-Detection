# tools/dataio.py
import joblib
import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y, add_channel=False):
        import numpy as np
        self.X = torch.tensor(X, dtype=torch.float32)
        if add_channel:  # for RNNs: (seq_len) -> (seq_len, 1)
            self.X = self.X.unsqueeze(-1)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

def load_splits(pkl_path):
    # must return: X_train, X_val, X_test, y_train, y_val, y_test, *_
    return joblib.load(pkl_path)

def make_loaders(X_train, X_val, y_train, y_val, *, batch_size=64, add_channel=False, num_workers=0):
    train_ds = TabularDataset(X_train, y_train, add_channel=add_channel)
    val_ds   = TabularDataset(X_val,   y_val,   add_channel=add_channel)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader
