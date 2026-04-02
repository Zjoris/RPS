import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import decode_image
import os

class RPS_Dataset(Dataset) :
    def __init__(self, labels_file_path : str, data_path : str) -> None:
        self._entries = pd.read_csv(labels_file_path, header = None)
        self._dataDir = data_path

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int ]:
        item_path = os.path.join(self._dataDir, str(self._entries.iloc[index, 0]))
        item = decode_image(item_path)
        label = self._entries.iloc[index, 1]
        return item, label