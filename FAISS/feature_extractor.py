import torch
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def extract_features(self, dataloader, is_inference: bool = False):
        all_features = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for batch in dataloader:
                if is_inference:
                    images, paths = batch
                    labels = None
                else:
                    images, labels, paths = batch

                images = images.to(self.device)
                feats = self.model(images).cpu().numpy()
                all_features.append(feats)
                if not is_inference:
                    all_labels.extend(labels)
                all_paths.extend(paths)

        features = np.vstack(all_features).astype('float32')
        if is_inference:
            return features, all_paths
        else:
            return features, all_labels, all_paths
