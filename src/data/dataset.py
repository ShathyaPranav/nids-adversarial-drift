"""PyTorch Dataset over the preprocessed .npz splits produced by preprocess.py."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"])
        self.y = torch.from_numpy(data["y"])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_meta(processed_dir: str) -> dict:
    with open(Path(processed_dir) / "meta.json") as f:
        return json.load(f)
