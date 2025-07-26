import torch
from torchvision import models
from typing import List
from utils.unfreeze_layer import unfreeze_model_layers

class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        unfreeze_layers: List = []
    ):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer, keep pooling
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]) 
        self.out_features = resnet.fc.in_features  # 2048 for resnet50

        # for name, param in self.feature_extractor.named_parameters():
        #     print(f" - {name}, requires grad = {param.requires_grad}")
        unfreeze_model_layers(self.feature_extractor, unfreeze_layers)

    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.feature_extractor(x)  # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        return features
