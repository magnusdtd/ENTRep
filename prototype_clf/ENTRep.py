import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Callable

class ENTRep(torch.nn.Module):
    def __init__(self, df: pd.DataFrame, transform:Callable=None):
        super().__init__()
        self.transform = transform
        self.df = df
        self.class_feature_map = {
          "nose-right": 0, 
          "nose-left" : 1, 
          "ear-right" : 2, 
          "ear-left"  : 3, 
          "vc-open"   : 4, 
          "vc-closed" : 5, 
          "throat"    : 6, 
        }

    def __len__(self):
        return len(self.df)

    def get_embeddings_for_class(self, id):
        
        class_idxs = self.df[self.df['Classification'] == id].index
        return self.df.iloc[class_idxs]['embedding']

    def __getitem__(self, idx: int):

        file_path = self.df["Path"].iloc[idx]
        if self.split != 'test':
            label = self.df["Classification"].iloc[idx]
        else:
            label = None

        image = Image.open(file_path)
    
        if self.transform:
            image = self.transform(image)
    
        if "embedding" in self.df.columns:
            emb = torch.tensor(self.df.iloc[idx]['embedding'], dtype=torch.float32).squeeze()
        else:
            emb = None
    
        return image, label, file_path, emb




