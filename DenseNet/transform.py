import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os

def get_transform(train:bool=True, inference:bool=False, image_size:Tuple[int, int]=(640, 480)):
    transforms_list = []
    if train:
        transforms_list.extend([
            # A.OneOf([
            #     A.Transpose(p=1.0),
            #     A.VerticalFlip(p=1.0),
            #     A.HorizontalFlip(p=1.0),
            # ], p=0.8),

            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),

            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2.0), p=1.0),
                A.GaussNoise(p=1.0)
            ], p=0.5),

            A.CoarseDropout(
                num_holes_range = (1, 3),
                p=0.4
            )
        ])
    transforms_list.extend([
        A.Resize(*image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return A.Compose(transforms_list)

def visualize_sample(dataloader: DataLoader, class_feature_map: dict):
    images, labels = next(iter(dataloader))
    batch_size = min(len(images), 9)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    inv_class_feature_map = {v: k for k, v in class_feature_map.items()}

    _, axes = plt.subplots(batch_size, 2, figsize=(10, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(batch_size):
        image_tensor = images[i]
        _class = int(labels["class"][i])
        img_name = labels["filename"][i]
        img_path = os.path.join('Dataset/train/imgs', img_name)
        img = Image.open(img_path).convert('RGB')

        class_name = inv_class_feature_map[_class]

        transformed_image = image_tensor.cpu() * std + mean
        transformed_image = transformed_image.permute(1, 2, 0).numpy()
        transformed_image = np.clip(transformed_image, 0, 1)

        transformed_image_resized = np.array(Image.fromarray((transformed_image * 255).astype(np.uint8)).resize(img.size)) / 255.0

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {img_name}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(transformed_image_resized)
        axes[i, 1].set_title(f"Transformed\nClass: {class_name} ({_class})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
