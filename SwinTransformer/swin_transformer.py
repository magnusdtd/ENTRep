import torch
from torch import nn
from classification.classification import Classification

class SwinTransformerCLS(Classification):
    def __init__(
        self,
        backbone,
        head_hidden_channel: int = 256,
        dropout_ratio: float = 0.4,
        lr: float = 1e-3,
        num_classes: int = 7,
        earlyStopping_patience: int = 7,
        criterion=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs=None
    ):
        super().__init__(num_classes, earlyStopping_patience)

        self.model = backbone.to(self.device)
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head.in_features, head_hidden_channel),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(head_hidden_channel, self.num_classes)
        ).to(self.device)

        # Loss function
        if criterion is None:
            self.classification_loss_fn = nn.CrossEntropyLoss().to(self.device)
        else:
            self.classification_loss_fn = criterion.to(self.device)

        # Optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': lr, 'weight_decay': 1e-4}
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        # Scheduler
        if scheduler_kwargs is None:
            scheduler_kwargs = {'mode': 'min', 'patience': 3}
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

    @staticmethod
    def load_model(model_path: str, backbone):
        model = SwinTransformerCLS(backbone=backbone)
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.model.eval()
        return model

    def load_state_dict(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
