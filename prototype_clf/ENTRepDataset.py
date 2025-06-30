import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ENTRepDataset(Dataset):
    def __init__(self, dataframe, transform=None, split='train'):
        self.df = dataframe
        self.transform = transform
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
            if isinstance(t, np.ndarray):
                self.emb_dim = t.shape[0] if t.ndim > 0 else 1
            elif isinstance(t, list):
                self.emb_dim = len(t)
            else:
                embeddings_array = np.array(self.df['embedding'].tolist())
                self.emb_dim = embeddings_array.shape[1] if embeddings_array.ndim > 1 else 1
        else:
            self.emb_dim = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df["Path"].iloc[idx] if "Path" in self.df.columns else None
        if self.split != 'test' and "Classification" in self.df.columns:
            label = self.df["Classification"].iloc[idx]
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = None

        image = Image.open(file_path) if file_path is not None else None
        if self.transform and image is not None:
            image = self.transform(image)

        emb = None
        if "embedding" in self.df.columns:
            emb = self.df.iloc[idx]['embedding']
            emb = torch.tensor(emb, dtype=torch.float32).squeeze()

        return image, label, file_path, emb

    def get_embeddings_for_class(self, class_id):
        class_idxs = self.df[self.df['Classification'] == class_id].index
        return self.df.iloc[class_idxs]['embedding'] 