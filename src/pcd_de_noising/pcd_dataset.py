from pathlib import Path

import h5py
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

DATA_KEYS = ["distance_1", "intensity_1", "distance_2",
             "intensity_2", "distance_3", "intensity_3"]
LABEL_KEY = "label"


class PCDDataset(Dataset):
    """HDF5 PyTorch Dataset to load distance, reflectivity, and labels from the PCD dataset.

    Input params:
        file_path: Path to the folder containing the dataset (1+ HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
    """

    def __init__(self, file_path, recursive):
        super().__init__()
        self.files = []

        p = Path(file_path)
        assert p.is_dir()

        self.files = sorted(p.glob("**/*.hdf5" if recursive else "*.hdf5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 files found")

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as h5_file:
            data = [h5_file[key][()] for key in DATA_KEYS]
            label = h5_file[LABEL_KEY][()]  # 128 * 1200

        data = tuple(torch.from_numpy(data) for data in data)
        data = torch.stack(data)  # 3 * 128 * 1200

        distance_1 = data[0:1, :, :]  # 1 * 128 * 1200
        reflectivity_1 = data[1:2, :, :]  # 1 * 128 * 1200
        distance_2 = data[2:3, :, :]  # 1 * 128 * 1200
        reflectivity_2 = data[3:4, :, :]  # 1 * 128 * 1200
        distance_3 = data[4:5, :, :]  # 1 * 128 * 1200
        reflectivity_3 = data[5:6, :, :]  # 1 * 128 * 1200

        label = torch.from_numpy(label).long()  # 128 * 1200
        # TODO: Discard 0s? Not going to learn anything useful from them
        #   Might teach the model that adverse weather isn't adverse weather,
        #   because it's labeled as nothing
        # label = torch.where(label == 0, torch.tensor(99), label)
        # label += 99

        # print(distance.shape, reflectivity.shape,
        #       second_reflectivity.shape, label.shape)

        assert (
            label.shape == distance_1.shape[1:]
        ), "Label shape does not match distance shape"
        assert (
            label.shape == reflectivity_1.shape[1:]
        ), "Label shape does not match reflectivity shape"

        return distance_1, reflectivity_1, distance_2, reflectivity_2, distance_3, reflectivity_3, label

    def __len__(self):
        return len(self.files)


class PointCloudDataModule(LightningDataModule):
    def __init__(self, train_directory, val_directory, test_directory):
        """Create a PointCloudDataModule

        Args:
            train_directory (str): path to the training hdf5 files
            val_directory (str): path to the validation hdf5 files
        """
        super().__init__()
        self.train_directory = train_directory
        self.val_directory = val_directory
        self.test_directory = test_directory

    def train_dataloader(self):
        dataset = PCDDataset(self.train_directory, recursive=True)
        print(f"Train found {len(dataset)} files")

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        return loader

    def val_dataloader(self):
        dataset = PCDDataset(self.val_directory, recursive=True)
        print(f"Val found {len(dataset)} files")

        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        return loader

    def test_dataloader(self):
        dataset = PCDDataset(self.test_directory, recursive=True)
        print(f"Test found {len(dataset)} files")

        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        return loader
