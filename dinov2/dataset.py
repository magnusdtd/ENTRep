from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from typing  import Callable

class ImageTextRetrievalDataset(Dataset):
    def __init__(
            self, 
            dataframe: pd.Dataframe, 
            processor, 
            tokenizer, 
            caption_column: str = 'Caption',
            path_column: str = 'Path',
            transform: Callable = None):
        self.dataframe = dataframe
        self.processor = processor
        self.tokenizer = tokenizer
        self.transform = transform
        self.caption_column = caption_column
        self.path_column = path_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx][self.path_column]
        text = self.dataframe.iloc[idx][self.caption_column]
        image = Image.open(img_path).convert("RGB")
        image=np.array(image)

        if self.transform:
            image = self.transform(image)['image']

        return {
            "image": image,
            "text": text
        }
