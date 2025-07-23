import torch
from PIL import Image
import pandas as pd
import numpy as np
from typing import Callable


class ENTRepDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer, 
        max_length: int,
        transform: Callable | None = None,
    ):
        self.df = df
        self.captions = df['Caption'].to_list()
        self.encoded_captions = tokenizer(
            self.captions, 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        img_path = self.df.iloc[idx]['Path']  
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform:
            transformed = self.transform(image=img)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0

        item['image'] = img_tensor
        item['caption'] = self.captions[idx]

        return item

