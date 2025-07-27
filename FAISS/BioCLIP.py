from FAISS.feature_extractor import FeatureExtractor
import torch
from PIL import Image
import open_clip
import numpy as np
from tqdm import tqdm

class BioCLIP_FE(FeatureExtractor):
    def __init__(
        self, 
        model_name: str, 
        model_path: str = "", 
    ):
        super().__init__()
        if model_path:
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
        else:
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name)

    def extract_features(self, dataloader, is_inference: bool = False):
        all_features = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting BioCLIP features"):
                if is_inference:
                    _, img_paths = batch
                    labels = None
                else:
                    _, labels, img_paths = batch

                for img_path in img_paths:
                    img = Image.open(img_path)
                    image_tensor = self.preprocess_val(img).unsqueeze(0).to(self.device)
                    image_feature = self.model.encode_image(image_tensor)
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    all_features.append(image_feature.cpu().numpy())
                if not is_inference:
                    all_labels.extend(labels)
                all_paths.extend(img_paths)

            features = np.vstack(all_features).astype('float32')
            if is_inference:
                return features, all_paths
            else:
                return features, all_labels, all_paths

    def extract_feature(self, img_path: str):
        """
        Extract feature from a single image at img_path
        """
        with torch.no_grad():
            img = Image.open(img_path)
            image_tensor = self.preprocess_val(img).unsqueeze(0).to(self.device)
            image_feature = self.model.encode_image(image_tensor)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            return image_feature.cpu().numpy()

    def extract_text_feature(self, text: str):
        """
        Extract feature from a single text string using BioCLIP
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_feature = self.model.encode_text(text_tokens)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            return text_feature.cpu().numpy()

    def extract_text_features(self, texts):
        """
        Extract features from a list of text strings using BioCLIP
        """
        all_features = []
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_features = text_features.cpu().numpy()
        return all_features

