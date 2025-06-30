import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ENTRepDataset(Dataset):
    def __init__(self, dataframe, transform=None, split='train', pair_mode=False):
        self.df = dataframe
        self.transform = transform
        self.split = split
        self.pair_mode = pair_mode
        if pair_mode:
            self.emb_dim = self.df['emb_dims'] if 'emb_dims' in self.df else None
        elif "embedding" in self.df.columns and len(self.df) > 0:
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
        if self.pair_mode:
            # For training: return both query and target images/embeddings
            row = self.df.iloc[idx]
            query_path = row['query_path']
            target_path = row['target_path']
            query_emb = torch.tensor(row['query_embedding'], dtype=torch.float32).squeeze()
            target_emb = torch.tensor(row['target_embedding'], dtype=torch.float32).squeeze()
            query_img = np.array(Image.open(query_path)) if query_path is not None else None
            target_img = np.array(Image.open(target_path)) if target_path is not None else None
            if self.transform and query_img is not None:
                query_img = self.transform(image=query_img)["image"]
            if self.transform and target_img is not None:
                target_img = self.transform(image=target_img)["image"]
            return query_img, target_img, query_emb, target_emb, query_path, target_path
        else:
            # For test/val: single image
            file_path = self.df["Path"].iloc[idx] if "Path" in self.df.columns else None
            image = np.array(Image.open(file_path)) if file_path is not None else None
            if self.transform and image is not None:
                image = self.transform(image=image)["image"]
            emb = None
            if "embedding" in self.df.columns:
                emb = self.df.iloc[idx]['embedding']
                emb = torch.tensor(emb, dtype=torch.float32).squeeze()
            return image, file_path, emb

    def get_embeddings_for_class(self, class_id):
        class_idxs = self.df[self.df['Classification'] == class_id].index
        return self.df.iloc[class_idxs]['embedding'] 
