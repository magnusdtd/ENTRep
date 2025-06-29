import torch
from torch import nn
from classification.classification import Classification

class ResNet(Classification):
    def __init__(
        self,
        backbone,
        hidden_channel: int = 256,
        dropout_ratio: float = 0.4,
        num_classes: int = 7,
        earlyStopping_patience: int = 7,
        criterion=None,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        use_mixup = False,
        mixup_alpha = 0.4,
        use_cutmix = False,
        cutmix_alpha = 1.0,
        adv_aug_prob = 0.5,
        **kwargs
    ):
        super().__init__(
            num_classes, 
            earlyStopping_patience,
            use_mixup,
            mixup_alpha,
            use_cutmix,
            cutmix_alpha,
            adv_aug_prob,
        )
        self.hidden_channel = hidden_channel
        self.model = backbone.to(self.device).float()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.hidden_channel),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(self.hidden_channel, self.num_classes)
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
        if optimizer == torch.optim.SGD:
            optimizer_kwargs.update({'momentum': 0.9})
        optimizer_kwargs.update(kwargs.get('optimizer_kwargs', {}))
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

        # Scheduler
        scheduler_kwargs = {}
        if scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler_kwargs.update({'mode': 'min', 'patience': 3})
        elif scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            scheduler_kwargs.update({'T_max': kwargs.get('T_max', 50)})
        scheduler_kwargs.update(kwargs.get('scheduler_kwargs', {}))
        self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)

    @staticmethod
    def load_model(model_path: str, backbone, hidden_channel: int):
        model = ResNet(backbone=backbone, hidden_channel=hidden_channel)
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.model.eval()
        return model

    def load_state_dict(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)