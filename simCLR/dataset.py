from torch.utils.data import Dataset
from typing import Callable
import pandas as pd
from PIL import Image
import numpy as np

class ENTRepDataset(Dataset):
    """
    Dataset for SimCLR-style training where each sample is a pair of images
    that should be close in representation space. The dataframe have
    columns 'Path1' and 'Path2', where each row defines a positive pair.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        transform: Callable
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path1 = self.df.iloc[idx]['Path1']
        img_path2 = self.df.iloc[idx]['Path2']

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        img1 = np.array(img1)
        img2 = np.array(img2)

        xi = self.transform(image=img1)["image"]
        xj = self.transform(image=img2)["image"]

        return xi, xj
