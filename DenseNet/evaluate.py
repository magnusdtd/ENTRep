import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

def evaluate_class_model(model, val_loader:DataLoader, class_feature_map:dict):
    """Evaluate the model on the validation dataset and plot a confusion matrix heatmap."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels['class'].to(device)

            classification_output, _ = model.forward(images)
            preds = torch.argmax(classification_output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = list(class_feature_map.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def evaluate_type_model(model, val_loader: DataLoader, type_feature_map: dict):
    """Evaluate the model on the validation dataset and plot a confusion matrix heatmap for type predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels['type'].to(device)

            _, type_output = model.forward(images)
            preds = torch.argmax(type_output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(type_feature_map.values()))
    type_names = list(type_feature_map.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=type_names, yticklabels=type_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap for Type Predictions')
    plt.show()

    print("Classification Report for Type Predictions:")
    print(classification_report(all_labels, all_preds, labels=list(type_feature_map.values()), target_names=type_names))