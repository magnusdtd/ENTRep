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
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        use_cutmix: bool = False,
        cutmix_alpha: float = 1.0,
        use_mosaic: bool = False,
        mosaic_alpha: float = 1.0,
        adv_aug_prob: float = 0.5,
        **kwargs
    ):
        self.model = backbone
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, hidden_channel),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channel, num_classes)
        )

        super().__init__(
            earlyStopping_patience=earlyStopping_patience,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha,
            use_mosaic=use_mosaic,
            adv_aug_prob=adv_aug_prob,
            **kwargs
        )

    @staticmethod
    def load_model(model_path: str, backbone, hidden_channel: int):
        model = ResNet(backbone=backbone, hidden_channel=hidden_channel)
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.model.eval()
        return model
