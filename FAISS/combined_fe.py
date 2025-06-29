from FAISS.BioCLIP import BioCLIP_FE
from FAISS.DINOv2 import DINOv2_FE
from FAISS.SAM_ViT import SAM_ViT_FE
import numpy as np
from FAISS.feature_extractor import FeatureExtractor

class CombinedFeatureExtractor(FeatureExtractor):
    def __init__(self, bioclip_args: dict, dinov2_args: dict, samvit_args: dict):
        self.bioclip = BioCLIP_FE(**bioclip_args)
        self.dinov2 = DINOv2_FE(**dinov2_args)
        self.samvit = SAM_ViT_FE(**samvit_args)

    def extract_feature(self, img_path:str):
        """
        Extract combined features from a single image at img_path
        """
        bio_feature = self.bioclip.extract_feature(img_path)
        dino_feature = self.dinov2.extract_feature(img_path)
        samvit_feature = self.samvit.extract_feature(img_path)
        
        # Concatenate features
        combined_feature = np.concatenate([bio_feature, dino_feature, samvit_feature], axis=-1)
        return combined_feature

    def extract_features(self, dataloader, is_inference:bool=False):

        if is_inference:
            bio_features, paths = self.bioclip.extract_features(dataloader, is_inference)
            dino_features, _ = self.dinov2.extract_features(dataloader, is_inference)
            samvit_features, _ = self.samvit.extract_features(dataloader, is_inference)
        else: 
            bio_features, labels, paths = self.bioclip.extract_features(dataloader, is_inference)
            dino_features, _, _ = self.dinov2.extract_features(dataloader, is_inference)
            samvit_features, _, _ = self.samvit.extract_features(dataloader, is_inference)

        # Concatenate features
        combined_features = np.concatenate([bio_features, dino_features, samvit_features], axis=-1)
        print(f'combined_features.shape = {bio_features.shape} + {dino_features.shape} + {samvit_features.shape} = {combined_features.shape}')

        if is_inference:
            return combined_features, paths
        else:
            return combined_features, labels, paths
