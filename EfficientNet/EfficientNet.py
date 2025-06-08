import torch
from torch import nn
from classification.classification import Classification
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EfficientNet(Classification):
  def __init__(
    self,
    backbone,
    lr: float = 1e-3,
    num_classes: int = 7,
    earlyStopping_patience: int = 7
  ):
    super().__init__(num_classes, earlyStopping_patience)

    self.model = backbone.to(self.device)
    self.model.classifier = nn.Sequential(
        nn.Linear(self.model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, self.num_classes)
    ).to(self.device)

    self.classification_loss_fn = nn.CrossEntropyLoss().to(self.device)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3)

  @staticmethod
  def load_model(model_path:str, backbone):
    model = EfficientNet(backbone=backbone)
    model.model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.model.eval()
    return model