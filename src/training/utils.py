import random
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(s: int = 42) -> None:
    """Set random seed for Python, NumPy, and torch."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class TabularSequenceDataset(Dataset):
    """Wrap tabular features so each row becomes a 1D sequence."""

    def __init__(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        self.X = torch.tensor(X_arr, dtype=torch.float32)
        y_arr = y.values if hasattr(y, "values") else y
        self.y = torch.tensor(y_arr, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return {"inputs": self.X[idx].unsqueeze(-1), "labels": self.y[idx]}
