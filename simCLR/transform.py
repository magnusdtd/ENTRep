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

