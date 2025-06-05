import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def load_model(model_path, device):
    """Load the trained model from the specified path."""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_image(image_path):
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
    outputs = model(image_tensor)
    classification_output = outputs[:, :model.num_classes]
    type_output = outputs[:, model.num_classes:]

    classification_prediction = torch.argmax(classification_output, dim=1).item()
    type_prediction = torch.argmax(type_output, dim=1).item()

    return classification_prediction, type_prediction

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to classify.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)
    image_tensor = preprocess_image(args.image_path)

    classification, type_ = classify_image(model, image_tensor, device)

    print(f"Classification Prediction: {classification}")
    print(f"Type Prediction: {type_}")
