import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torchvision.io import decode_image
# from torchvision import models
import os

class RPS_Dataset(Dataset) :
    def __init__(self, labels_file_path : str, data_path : str) -> None:
        self._entries = pd.read_csv(labels_file_path, header = None)
        self._dataDir = data_path

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item_path = os.path.join(self._dataDir, str(self._entries.iloc[index, 0]))
        item = decode_image(item_path)
        label = self._entries.iloc[index, 1]
        return item, label
    
class RPS_Classifier_torch(nn.Module): 
    def __init__(self, n_classes : int, channels : int = 3, dims : tuple[int, int] = (600, 600)) -> None:
        super().__init__()
        x = int(dims[0]/2/2 - 3)
        y = int(dims[1]/2/2 - 3)
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * x * y, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, item : torch.Tensor): 
        x = self.pool(F.relu(self.conv1(item)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
