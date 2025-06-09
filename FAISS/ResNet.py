from FAISS.feature_extractor import FeatureExtractor
import torch.nn as nn
from torchvision import models
import torch

class ResNet_FE(FeatureExtractor):
  def __init__(self, backbone):
    super().__init__()
    self.model = backbone
    self.model.fc = nn.Identity()
    self.model.to(self.device)
    self.model.eval()

  @staticmethod
  def load_model_state(model_path: str, backbone):
    model = ResNet_FE(backbone=backbone)
    state_dict = torch.load(model_path, map_location=model.device)
    
    # Filter out keys related to the fully connected layer
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

    model.model.load_state_dict(filtered_state_dict, strict=False)
    model.model.eval()
    return model
