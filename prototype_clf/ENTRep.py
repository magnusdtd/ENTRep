import torch
from PIL import Image
import pandas as pd
from typing import Callable
import numpy as np

class ENTRep(torch.nn.Module):
    def __init__(self, df: pd.DataFrame, transform:Callable=None, split:str='train'):
        super().__init__()
        self.transform = transform
        self.df = df
        self.split = split
        self.class_feature_map = {
            "nose-right": 0, 
            "nose-left" : 1, 
            "ear-right" : 2, 
            "ear-left"  : 3, 
            "vc-open"   : 4, 
            "vc-closed" : 5, 
            "throat"    : 6, 
        }
        
        self.n_classes = len(self.class_feature_map)
        if "embedding" in self.df.columns and len(self.df) > 0:
            t = self.df['embedding'].iloc[0]

            # Handle different types of embeddings
            if isinstance(t, np.ndarray):
                self.emb_dim = t.shape[0] if t.ndim > 0 else 1
            elif isinstance(t, list):
                self.emb_dim = len(t)
            else:
                embeddings_array = np.array(self.df['embedding'].tolist())
                self.emb_dim = embeddings_array.shape[1] if embeddings_array.ndim > 1 else 1
        else:
            self.emb_dim = None

        print(f'self.emb_dim = {self.emb_dim}')

    def __len__(self):
        return len(self.df)

    def get_embeddings_for_class(self, id):
        
        class_idxs = self.df[self.df['Classification'] == id].index
        return self.df.iloc[class_idxs]['embedding']

    def __getitem__(self, idx: int):

        file_path = self.df["Path"].iloc[idx]
        if self.split != 'test':
            label = self.df["Classification"].iloc[idx]
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = None

        image = Image.open(file_path)
    
        if self.transform:
            image = self.transform(image)
    
        if "embedding" in self.df.columns:
            emb = self.df.iloc[idx]['embedding']
            print(f'emb type = {type(emb)}, shape = {emb.shape if hasattr(emb, "shape") else "no shape"}')
            
            emb = torch.tensor(emb, dtype=torch.float32)
            
            print(f'emb type = {type(emb)}, shape = {emb.shape}')
            emb = emb.squeeze()
            print(f'emb after squeeze type = {type(emb)}, shape = {emb.shape}')

        else:
            emb = None
    
        return image, label, file_path, emb
