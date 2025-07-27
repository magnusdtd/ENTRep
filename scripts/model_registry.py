import torchvision.models as models
from resnet.resnet import ResNet
from DenseNet.DenseNet import DenseNet
from EfficientNet.EfficientNet import EfficientNet
from SwinTransformer.swin_transformer import SwinTransformer

# Model registry for dynamic selection
MODEL_REGISTRY = {
    "ResNet50": {
        "class": ResNet,
        "backbone": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "hidden_channel": 512,
    },
    "DenseNet121": {
        "class": DenseNet,
        "backbone": lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "hidden_channel": 512,
    },
    "EfficientNet_B0": {
        "class": EfficientNet,
        "backbone": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "hidden_channel": 512,
    },
    "Swin_T": {
        "class": SwinTransformer,
        "backbone": lambda: models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
        "hidden_channel": 512,
    },
}
