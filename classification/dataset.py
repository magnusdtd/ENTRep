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
            img_path: str = 'Dataset/train/imgs', 
            transform: callable = None
        ):
        self.df = df
        self.transform = transform
        self.class_feature_map = class_feature_map
        self.img_path = img_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int):
        img_name = self.df.iloc[idx]['Path']  
        img_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        label = {
            'class': torch.tensor(self.class_feature_map[self.df.iloc[idx]['Classification']], dtype=torch.long),
            'filename': img_name
        }
        
        if self.transform:
            transformed = self.transform(
                image=img
            )
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, label