import torch
from typing import Callable, Dict
import tqdm
from PIL import Image
import json
import numpy as np
import pandas as pd      

class DinoV2:
    def __init__(self, backbone, img_folder_path:str, transform: Callable):
        self.model = backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.transform = transform
        self.img_folder_path = img_folder_path
        
    def load_img(self, img_path:str) -> torch.Tensor:
        img = Image.open(img_path)

        transformed_img = self.transform(img)[:3].unsqueeze(0)

        return transformed_img
    
    def compute_embeddings(self, file_path:str) -> Dict:
        files = []
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None, names=['Query'])
            files = df['Query'].to_list()
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, typ='series')
            df = df.reset_index()
            df.columns = ['Filename', 'Label']
            df['Filename'] = df['Filename'].apply(lambda x: f"{self.img_folder_path}/{x}")
            files = df['Filename'].tolist()
        else:
            raise ValueError("Unsupported file type. Only CSV and JSON are supported.")
        

        embeddings = {}

        with torch.no_grad():
            for _, file in enumerate(tqdm.tqdm(files)):
                output = self.model(self.load_img(file).to(self.device))

                embeddings[file] = np.array(output[0].cpu().numpy()).reshape(1, -1).tolist()

        with open("embeddings.json", "w") as f:
            f.write(json.dumps(embeddings))

        return embeddings
