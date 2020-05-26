import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset


class MoleDataset(Dataset):
    """Mole dataset."""

    def __init__(self, input_folder, csv_file, transform=None):
        """
        Args:
            input_folder (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._metadata = pd.read_csv(csv_file)
        self._input_folder = input_folder
        self._transform = transform

        self._images = os.listdir(input_folder)
        self._names = self._metadata.dx.unique().tolist()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self._images[idx]
        img_path = os.path.join(self._input_folder, img_name)
        image = cv2.imread(img_path)
        if self._transform:
            image = self._transform(image)

        dx = self._metadata.loc[self._metadata['image_id'] == img_name.split(".")[0]].iloc[0]["dx"]
        label = self._names.index(dx)

        return image, label
