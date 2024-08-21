import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NPYDataset(Dataset):
    def __init__(self, npy_files, augmentor=None):
        self.data_files = npy_files
        self.augmentor = augmentor

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

        # data augement
        if self.augmentor:
            pre_input, pre_label = self.augmentor(pre_input, pre_label)

        input_data = torch.tensor(pre_input, dtype=torch.float32)  # 前4个维度作为输入
        label = torch.tensor(pre_label, dtype=torch.long)  # 最后1个维度作为标签
        return input_data, label

    # def __getitem__(self, idx):
    #     # get idx and idx+1 data
    #     idx_next = idx + 1 if idx + 1 < len(self.data_files) else 0

    #     data_path = self.data_files[idx]
    #     data = np.load(data_path)
    #     data_next_path = self.data_files[idx_next]
    #     data_next = np.load(data_next_path)

    #     # [x y z range pluse] * 2 + label
    #     pre_input = data[[0, 1, 2], :, :]


def create_dataloader(data_dir_list, batch_size=4):
    # data_dir_list
    npy_files = []
    for data_dir in data_dir_list:
        current_files = [os.path.join(data_dir, f)
                         for f in os.listdir(data_dir) if f.endswith('.npy')]
        npy_files.extend(current_files)
    dataset = NPYDataset(npy_files, augmentor=DataAugment())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class DataAugment:
    def __init__(self, translation_range=(-10, 10), rotation_range=(-np.pi, np.pi)):
        self.translation_range = translation_range
        self.rotation_range = rotation_range

    def random_translation(self):
        tx = np.random.uniform(
            self.translation_range[0], self.translation_range[1])
        ty = np.random.uniform(
            self.translation_range[0], self.translation_range[1])
        tz = np.random.uniform(
            self.translation_range[0], self.translation_range[1])
        translation_matrix = np.array([tx, ty, tz])
        return translation_matrix

    def random_rotation(self):
        angle = np.random.uniform(
            self.rotation_range[0], self.rotation_range[1])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rotate around the Z axis (roll direction)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        return rotation_matrix

    def random_augmentation(self, points):
        R = self.random_rotation()
        T = self.random_translation()

        points.reshape(-1, 128, 1200)
        for i in range(points.shape[1]):
            for j in range(points.shape[2]):
                point = points[:, i, j]
                if point[0] == 0 and point[1] == 0 and point[2] == 0:
                    continue
                else:
                    point = np.dot(R, point) + T
                    points[:, i, j] = point
        return points

    def __call__(self, input_data, label):
        # input_data = self.random_augmentation(input_data)
        seed = np.random.randint(0,4)
        if seed == 1:
            input_data = np.flip(input_data,axis=2).copy()
            label = np.flip(label, axis=1).copy()
        elif seed == 2:
            input_data = np.flip(input_data,axis=1).copy()
            label = np.flip(label, axis=0).copy()
        elif seed == 3:
            input_data = np.flip(input_data,axis=2).copy()
            label = np.flip(label, axis=1).copy()
            input_data = np.flip(input_data,axis=1).copy()
            label = np.flip(label, axis=0).copy()
        return input_data, label
