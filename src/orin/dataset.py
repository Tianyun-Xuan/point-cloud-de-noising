import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NPYDataset(Dataset):
    def __init__(self, npy_files):
        self.data_files = npy_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = np.load(data_path)
        input_data = torch.tensor(data[:6], dtype=torch.float32)  # 前6个维度作为输入
        label = torch.tensor(data[6], dtype=torch.long)  # 最后1个维度作为标签
        return input_data, label

def create_dataloader(data_dir, batch_size=4):
    npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    dataset = NPYDataset(npy_files)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
