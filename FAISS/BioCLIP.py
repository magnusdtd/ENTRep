from FAISS.feature_extractor import FeatureExtractor
import torch
from PIL import Image
import open_clip
import numpy as np
import os

class BioCLIP_FE(FeatureExtractor):
  def __init__(
      self, 
      model_name: str, 
      model_path:str="", 
      img_folder_path:str = 'Dataset/train/imgs'
    ):
    super().__init__()
    if model_path:
      self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
    else:
      self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
    self.model.to(self.device)
    self.model.eval()
    self.img_folder_path = img_folder_path

  def extract_features(self, dataloader, is_inference: bool = False):
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
      for batch in dataloader:
        if is_inference:
          _, img_names = batch
          labels = None
        else:
          _, labels, img_names = batch

        for img_name in img_names:
          img_path = os.path.join(self.img_folder_path, img_name)
          img = Image.open(img_path)

          image_tensor = self.preprocess_val(img).unsqueeze(0).to(self.device)

          image_feature = self.model.encode_image(image_tensor)
          image_feature /= image_feature.norm(dim=-1, keepdim=True)

          all_features.append(image_feature.cpu().numpy())
        if not is_inference:
          all_labels.extend(labels)
        all_paths.extend(img_names)

      features = np.vstack(all_features).astype('float32')
      if is_inference:
        return features, all_paths
      else:
        return features, all_labels, all_paths
