import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

def evaluate_model(
        model, 
        val_loader:DataLoader, 
        label_encoder:dict, 
        cm_path="confusion_matrix.png", 
        report_path="classification_report.txt"
    ):
    """Evaluate the model on the validation dataset, plot and save a confusion matrix heatmap, and save the classification report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels['class']
            labels = labels.to(device)

            outputs = model.forward(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = list(label_encoder.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.show()

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    with open(report_path, "w") as f:
        f.write(report)
