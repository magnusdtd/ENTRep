import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision.models as models
from DenseNet.DenseNet import DenseNet
import os, random
import pandas as pd
import matplotlib.pyplot as plt

def load_model(model_path:str, backbone, device:str):
    """Load the trained model state dictionary into the DenseNet architecture."""
    model = DenseNet(backbone=backbone)
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    model.model.eval()
    return model

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
    df:pd.DataFrame, 
    class_feature_map:dict, 
    device: str,
    image_folder: str = "Dataset/train/imgs"
):
    """Randomly select 9 images from the folder"""

    inv_class_feature_map = {v: k for k, v in class_feature_map.items()}
    image_files = os.listdir(image_folder)
    selected_images = random.sample(image_files, 9)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        image_path = os.path.join(image_folder, selected_images[i])
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.basename(image_path)
        true_class = df[df['Path'] == image_name]['Classification']
        if true_class.empty:
            print(f"Warning: No matching entry found for {image_name} in DataFrame.")
            continue
        true_class = true_class.values[0]
        image_tensor = preprocess_image(image_path)
        class_ = classify_image(model, image_tensor, device)

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(
            f"True: {true_class}\nPred: {inv_class_feature_map[class_]}",
        )
        fig.suptitle("True Labels and Predictions for Random Images", fontsize=18, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to classify.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)
    image_tensor = preprocess_image(args.image_path)

    classification = classify_image(model, image_tensor, device)

    print(f"Classification Prediction: {classification}")
