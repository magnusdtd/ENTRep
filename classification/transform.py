import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd

def get_transform(train: bool = True, img_size: Tuple[int, int] = (480, 640)):
    transforms_list = []
    # transforms_list.extend([
    #     A.LongestMaxSize(max_size=img_size),
    #     A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
    # ])
    if train:
        transforms_list.extend([

            A.Affine(translate_percent=0.05, scale=(0.8, 1.2), rotate=0, p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            A.RandomBrightnessContrast(p=0.3),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(p=1.0)
            ], p=0.5),

            A.CoarseDropout(num_holes_range=(1, 5), p=0.5),
        ])
    transforms_list.extend([
        A.Resize(*img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return A.Compose(transforms_list)

def visualize_sample(df: pd.DataFrame, dataloader: DataLoader, label_encoder: dict):
    images, labels = next(iter(dataloader))
    batch_size = min(len(images), 9)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    inv_label_encoder = {v: k for k, v in label_encoder.items()}

    _, axes = plt.subplots(batch_size, 2, figsize=(10, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(batch_size):
        image_tensor = images[i]
        _class = int(labels["class"][i])
        img_path = labels["img_path"][i]
        img = Image.open(img_path).convert('RGB')

        class_name = inv_label_encoder[_class]

        # Use df to find the original label for this image
        # df has columns "Path" and "Label"
        # Find the row where "Path" matches img_path
        orig_label_row = df[df["Path"] == img_path]
        if not orig_label_row.empty:
            orig_label = orig_label_row.iloc[0]["Label"]
        else:
            orig_label = "Unknown"

        transformed_image = image_tensor.cpu() * std + mean
        transformed_image = transformed_image.permute(1, 2, 0).numpy()
        transformed_image = np.clip(transformed_image, 0, 1)

        transformed_image_resized = np.array(Image.fromarray((transformed_image * 255).astype(np.uint8)).resize(img.size)) / 255.0

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {img_path}\nOriginal Label: {orig_label}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(transformed_image_resized)
        axes[i, 1].set_title(f"Transformed\nClass: {class_name} ({_class})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
