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
        # [x y z range pluse] * 2 + label

        pre_input = data[[0, 1, 2], :, :]
        pre_label = data[-1, :, :]
        pre_label[pre_label == 2] = 0
        pre_label[pre_label == 3] = 1

        input_data = torch.tensor(pre_input, dtype=torch.float32)  # 前4个维度作为输入
        label = torch.tensor(pre_label, dtype=torch.long)  # 最后1个维度作为标签
        return input_data, label


def create_dataloader(data_dir_list, batch_size=4):
    # data_dir_list
    npy_files = []
    for data_dir in data_dir_list:
        current_files = [os.path.join(data_dir, f)
                         for f in os.listdir(data_dir) if f.endswith('.npy')]
        npy_files.extend(current_files)
    dataset = NPYDataset(npy_files)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
