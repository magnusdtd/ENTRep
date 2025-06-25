from FAISS.BioCLIP import BioCLIP_FE
from FAISS.DINOv2 import DINOv2_FE
import numpy as np
from FAISS.feature_extractor import FeatureExtractor

class CombinedFeatureExtractor(FeatureExtractor):
    def __init__(self, bioclip_args: dict, dinov2_args: dict):
        self.bioclip = BioCLIP_FE(**bioclip_args)
        self.dinov2 = DINOv2_FE(**dinov2_args)
        self.device = self.bioclip.device

    def extract_features(self, dataloader, is_inference:bool=False):

        if is_inference:
            bio_features, paths = self.bioclip.extract_features(dataloader, is_inference)
            dino_features, _ = self.dinov2.extract_features(dataloader, is_inference)
        else: 
            bio_features, labels, paths = self.bioclip.extract_features(dataloader, is_inference)
            dino_features, _, _ = self.dinov2.extract_features(dataloader, is_inference)

        # Concatenate features
        combined_features = np.concatenate([bio_features, dino_features], axis=1)
        
        if is_inference:
            return combined_features, paths
        else:
            return combined_features, labels, paths
