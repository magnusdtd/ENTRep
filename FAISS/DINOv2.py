from FAISS.feature_extractor import FeatureExtractor
import torch
from PIL import Image
import numpy as np
import os
from typing import Tuple
import torchvision.transforms as T

class DINOv2_FE(FeatureExtractor):
  def __init__(
      self, 
      repo_name: str = 'facebookresearch/dinov2', 
      model_name:str='dinov2_vits14',
      img_folder_path:str = 'Dataset/train/imgs',
      image_size: Tuple[int, int] = (490, 644)
    ):
    super().__init__()
    self.model = torch.hub.load(repo_name, model_name)
    self.model.to(self.device)
    self.model.eval()
    self.img_folder_path = img_folder_path
    self.image_size = image_size
    
    self.preprocess = T.Compose([
        T.Resize(image_size, interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def load_img(self, img_path:str) -> torch.Tensor:
    img = Image.open(img_path)
    
    if img.mode != 'RGB':
      img = img.convert('RGB')
    
    transformed_img = self.preprocess(img).unsqueeze(0)
    
    return transformed_img

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

          output = self.model(self.load_img(img_path).to(self.device))

          all_features.append(np.array(output[0].cpu().numpy()).reshape(1, -1).tolist())
          
        if not is_inference:
          all_labels.extend(labels)
        all_paths.extend(img_names)

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
      output = self.model(self.load_img(img_path).to(self.device))
      return output[0].cpu().numpy() 