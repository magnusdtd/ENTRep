import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class ENTRepDataset(Dataset):
    def __init__(
      self, 
      df: pd.DataFrame, 
      class_feature_map:dict, 
      transform: callable = None, 
      is_inference: bool = False
    ):
        self.df = df
        self.transform = transform
        self.class_feature_map = class_feature_map
        self.is_inference = is_inference

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) \
        -> tuple[torch.Tensor, torch.Tensor, str] | tuple[torch.Tensor, str]:
        img_path = self.df.iloc[idx]['Path']
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform:
            transformed = self.transform(
                image=img
            )
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0

        if self.is_inference:
            return img_tensor, img_path
        
        class_name = self.df.iloc[idx]['Classification']
        label = torch.tensor(
            self.class_feature_map[class_name], dtype=torch.long
        )

        return img_tensor, label, img_path