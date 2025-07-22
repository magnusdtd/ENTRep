import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from typing import Callable, Dict
import cv2
import uuid

class ENTRepDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            label_encoder: Dict, 
            transform: Callable | None = None,
            is_train: bool = False
        ):
        self.df = df
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_train = is_train
        self.inv_label_encoder = {v: k for k, v in label_encoder.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int):
        img_path = self.df.iloc[idx]['Path']  
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        label = {
            'class': torch.tensor(self.label_encoder[self.df.iloc[idx]['Label']], dtype=torch.long),
            'img_path': img_path
        }
        
        if self.transform:
            transformed = self.transform(image=img)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0

        if self.is_train and np.random.rand() < 0.5:
            img_tensor = torch.flip(img_tensor, dims=[-1])
            original_label = self.inv_label_encoder[int(label['class'])]
            temp = original_label
            if original_label == "nose-right":
                temp = "nose-left"
            elif original_label == "nose-left":
                temp = "nose-right"
            elif original_label == "ear-right":
                temp = "ear-left"
            elif original_label == "ear-left":
                temp = "ear-right"
            label['class'] = torch.tensor(self.label_encoder[temp], dtype=torch.long)

        return img_tensor, label