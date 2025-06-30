from FAISS.BioCLIP import BioCLIP_FE
from FAISS.DINOv2 import DINOv2_FE
from FAISS.SAM_ViT import SAM_ViT_FE
import numpy as np
from FAISS.feature_extractor import FeatureExtractor

class CombinedFeatureExtractor(FeatureExtractor):
    def __init__(
            self, 
            bioclip_args: dict = None, 
            dinov2_args: dict = None, 
            samvit_args: dict = None
    ):
        if not (bioclip_args or dinov2_args or samvit_args):
            raise ValueError("At least one model args (bioclip_args, dinov2_args, samvit_args) must be provided.")
        self.bioclip = BioCLIP_FE(**bioclip_args) if bioclip_args is not None else None
        self.dinov2 = DINOv2_FE(**dinov2_args) if dinov2_args is not None else None
        self.samvit = SAM_ViT_FE(**samvit_args) if samvit_args is not None else None

    def extract_feature(self, img_path: str):
        """
        Extract combined features from a single image at img_path
        """
        features = []
        if self.bioclip is not None:
            bio_feature = self.bioclip.extract_feature(img_path)
            features.append(bio_feature)
        if self.dinov2 is not None:
            dino_feature = self.dinov2.extract_feature(img_path)
            features.append(dino_feature)
        if self.samvit is not None:
            samvit_feature = self.samvit.extract_feature(img_path)
            features.append(samvit_feature)
        if not features:
            raise ValueError("No feature extractors are initialized.")
        combined_feature = np.concatenate(features, axis=-1)
        return combined_feature

    def extract_features(self, dataloader, is_inference: bool = False):
        '''
        Extract combined features for a batch of images.
        '''
        features = []
        labels = None
        paths = None

        if is_inference:
            if self.bioclip is not None:
                bio_features, paths = self.bioclip.extract_features(dataloader, is_inference)
                features.append(bio_features)
            if self.dinov2 is not None:
                dino_features, _ = self.dinov2.extract_features(dataloader, is_inference)
                features.append(dino_features)
            if self.samvit is not None:
                samvit_features, _ = self.samvit.extract_features(dataloader, is_inference)
                features.append(samvit_features)
            if not features:
                raise ValueError("No feature extractors are initialized.")
            combined_features = np.concatenate(features, axis=-1)
            return combined_features, paths
        else:
            if self.bioclip is not None:
                bio_features, labels, paths = self.bioclip.extract_features(dataloader)
                features.append(bio_features)
            if self.dinov2 is not None:
                dino_features, _, _ = self.dinov2.extract_features(dataloader)
                features.append(dino_features)
            if self.samvit is not None:
                samvit_features, _, _ = self.samvit.extract_features(dataloader)
                features.append(samvit_features)
            if not features:
                raise ValueError("No feature extractors are initialized.")
            combined_features = np.concatenate(features, axis=-1)
            print(f'combined_features.shape = {[f.shape for f in features]} = {combined_features.shape}')
            return combined_features, labels, paths
