import os
import random
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset


class ClassifierDataLoader(Dataset):
    """
    A dataloader for Classifier Network
    """

    def __init__(self, base_dir, df_path, transformation=None):
        """
        Parameters :
            - base_dir: path to base directory of data
            - df_path: path to the dataframe of data
            - transformation: torchvision.transforms
        """

        super().__init__()

        self.base_dir = base_dir
        self.transformation = transformation

        self.df = pd.read_csv(df_path)
        self.df.drop(columns="Unnamed: 0", inplace=True)

        self.df["path"] = self.df.apply(
            lambda x: os.path.join(base_dir, x[0], x[2]), axis=1)

        self.images = list(zip(self.df["path"], self.df["id"]))
        random.shuffle(self.images)

        self.num_categories = len(self.df["id"].unique())

    def __getitem__(self, index):
        """
        In this function, an image and its one-hot label is returned.
        """

        img_path, img_cat = self.images[index]

        x = torchvision.io.read_image(img_path) / 255
        if self.transformation is not None:
            x = self.transformation(x)

        y = torch.zeros(self.num_categories)
        y[img_cat] = 1

        return x, y

    def __len__(self):
        """
        `len(.)` function return number of data in dataset
        """
        return len(self.images)
