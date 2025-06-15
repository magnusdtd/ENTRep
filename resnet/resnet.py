import torch
from torch import nn
from classification.classification import Classification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.focal_loss import FocalLoss

class ResNet(Classification):
    def __init__(
        self,
        backbone,
        head_hidden_channel: int = 256,
        dropout_ratio: float = 0.4,
        num_classes: int = 7,
        earlyStopping_patience: int = 7,
        criterion=None,
        optimizer_class=torch.optim.Adam,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        **kwargs
    ):
        super().__init__(num_classes, earlyStopping_patience)

        self.model = backbone.to(self.device).float()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, head_hidden_channel),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(head_hidden_channel, self.num_classes)
        ).to(self.device).float()

        # Loss function
        if criterion is None:
            class_weights = kwargs.get('class_weights', None)
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, device=self.device).float()
            self.classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(self.device).float()
        else:
            self.classification_loss_fn = criterion.to(self.device).float()

        # Optimizer
        optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 1e-4}
        optimizer_kwargs.update(kwargs.get('optimizer_kwargs', {}))
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        # Scheduler
        scheduler_kwargs = {'mode': 'min', 'patience': 3}
        scheduler_kwargs.update(kwargs.get('scheduler_kwargs', {}))
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

    @staticmethod
    def load_model(model_path: str, backbone):
        model = ResNet(backbone=backbone)
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.model.eval()
        return model

    def load_state_dict(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)