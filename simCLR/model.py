import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    resnet50,
    densenet121,
    efficientnet_b0,
    swin_t,
)


class SimCLR(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=128):
        super().__init__()
        self.base_model = base_model.lower()
        if self.base_model == 'resnet50':
            self.encoder = resnet50(pretrained=False)
            dim_mlp = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            self._encode = lambda x: self.encoder(x)
        elif self.base_model == 'densenet121':
            self.encoder = densenet121(pretrained=False)
            dim_mlp = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
            self._encode = lambda x: self.encoder(x)
        elif self.base_model == 'efficientnetb0' or self.base_model == 'efficientnet_b0':
            self.encoder = efficientnet_b0(pretrained=False)
            dim_mlp = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
            self._encode = lambda x: self.encoder(x)
        elif self.base_model == 'swintransformer' or self.base_model == 'swin_t':
            self.encoder = swin_t(pretrained=False)
            dim_mlp = self.encoder.head.in_features
            self.encoder.head = nn.Identity()
            self._encode = lambda x: self.encoder(x)
        else:
            raise ValueError(f"Unsupported base_model: {base_model}")

        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        h = self._encode(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def encode(self, x):
        """Return the encoder output (before projection head)"""
        h = self._encode(x)

    def load_pretrained(self, weight_path, map_location="cpu"):
        """
        Load pretrained weights from disk.
        """
        state_dict = torch.load(weight_path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weight_path}")
