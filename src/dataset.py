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
        # # print("data.shape: ", data.shape)
        # data[0, :, :] = data[0, :, :] / 64
        # data[2, :, :] = data[2, :, :] / 64
        # control the range of the data to [0, 1023]
        # data[0, :, :] = np.clip(data[0, :, :], 0, 1023)
        # data[2, :, :] = np.clip(data[2, :, :], 0, 1023)
        # print("echo 1 range : [{}, {}]".format(
        #     np.min(data[0, :, :]), np.max(data[0, :, :])))
        # print("echo 1 pluse : [{}, {}]".format(
        #     np.min(data[1, :, :]), np.max(data[1, :, :])))
        # print("echo 2 range : [{}, {}]".format(
        #     np.min(data[2, :, :]), np.max(data[2, :, :])))
        # print("echo 2 pluse : [{}, {}]".format(
        #     np.min(data[3, :, :]), np.max(data[3, :, :])))
        input_data = torch.tensor(data[:4], dtype=torch.float32)  # 前4个维度作为输入
        label = torch.tensor(data[4], dtype=torch.long)  # 最后1个维度作为标签
        return input_data, label


def create_dataloader(data_dir, batch_size=4):
    npy_files = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith('.npy')]
    dataset = NPYDataset(npy_files)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
