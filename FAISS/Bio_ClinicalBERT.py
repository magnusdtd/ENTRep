import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from FAISS.feature_extractor import FeatureExtractor

class Bio_ClinicalBERT(FeatureExtractor):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_feature(self, text: str):
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embedding = cls_embedding.squeeze(0).cpu().numpy()
            return embedding

    def extract_features(self, texts, is_inference: bool = False):
        all_features = []
        for text in texts:
            emb = self.extract_feature(text)
            all_features.append(emb)
        features = np.vstack(all_features).astype('float32')
        return features
