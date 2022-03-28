import os
import random
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset


class VerifierDataLoader(Dataset):
    """
    A dataloader for Verifier Network
    """

    def __init__(self, base_dir, csv_path, transformation) -> None:
        """
            Parameters:
                - base_dir (str): path to base directory
                - csv_path (str): path to csv file 
        """
        super().__init__()

        self.transformation = transformation

        self.df = pd.read_csv(csv_path)

        self.df["image_A"] = self.df["image_A"].apply(
            lambda path: os.path.join(base_dir, path))
        self.df["image_B"] = self.df["image_B"].apply(
            lambda path: os.path.join(base_dir, path))

        self.df = self.df.sample(frac=1)

    def __getitem__(self, index):
        x1_path = self.df.loc[index, "image_A"]
        x2_path = self.df.loc[index, "image_B"]
        y = self.df.loc[index, "match"]

        x1 = torchvision.io.read_image(x1_path) / 255
        x2 = torchvision.io.read_image(x2_path) / 255

        if self.transformation is not None:
            x1 = self.transformation(x1)
            x2 = self.transformation(x2)

        return x1, x2, y

    def __len__(self):
        return self.df.shape[0]
