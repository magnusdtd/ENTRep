import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os, random
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_image(image_path:str):
    """Preprocess the image for inference using Albumentations."""
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0)

def classify_image(model, image_tensor, device):
    """Classify the image using the model."""
    image_tensor = image_tensor.to(device)
    outputs = model.forward(image_tensor)

    classification_prediction = torch.argmax(outputs, dim=1).item()

    return classification_prediction

def random_inference_9_images(
    model, 
    df: pd.DataFrame, 
    label_encoder: dict, 
    device: str,
):
    """Randomly select 9 images from the DataFrame and display predictions."""

    inv_label_encoder = {v: k for k, v in label_encoder.items()}

    # Randomly sample 9 rows from the DataFrame
    if len(df) < 9:
        raise ValueError("DataFrame has fewer than 9 images.")
    sampled_df = df.sample(n=9, random_state=None).reset_index(drop=True)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        image_path = sampled_df.loc[i, 'Path']
        true_label = sampled_df.loc[i, 'Label']
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist.")
            ax.axis('off')
            ax.set_title(f"Missing: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_image(image_path)
        class_ = classify_image(model, image_tensor, device)

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(
            f"True: {true_label}\nPred: {inv_label_encoder[class_]}",
        )
        fig.suptitle("True Labels and Predictions for Random Images", fontsize=18, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
