import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Resnet:
  def __init__(self, backbone, num_classes: int = 7, num_types: int = 2):
    self.num_classes = num_classes
    self.num_types = num_types

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model = backbone.to(self.device)
    self.model.fc = nn.Sequential(
      nn.Linear(self.model.fc.in_features, 256),
      nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(256, self.num_classes + self.num_types)
    ).to(self.device)

    self.classification_loss_fn = nn.CrossEntropyLoss().to(self.device)
    self.type_loss_fn = nn.CrossEntropyLoss().to(self.device)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    self.epochs = 0

    self.train_losses = []
    self.val_losses = []
    self.classification_accuracies = []
    self.type_accuracies = []

  def forward(self, images: torch.Tensor):
    images = images.to(self.device)
    outputs = self.model(images)
    classification_output = outputs[:, :self.num_classes]
    type_output = outputs[:, self.num_classes:]
    return classification_output, type_output
  
  def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
    self.epochs = epochs

    for epoch in range(self.epochs):
      self.model.train()
      train_loss = 0

      num_batches = len(train_loader)
      num_images = len(train_loader.dataset)
      train_progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch + 1}")

      for _, (images, labels) in train_progress_bar:
        images = images.to(self.device)
        labels['class'] = labels['class'].to(self.device)
        labels['type'] = labels['type'].to(self.device)

        self.optimizer.zero_grad()

        classification_output, type_output = self.forward(images)
        classification_loss = self.classification_loss_fn(classification_output, labels['class'])
        type_loss = self.type_loss_fn(type_output, labels['type'])

        loss = classification_loss + type_loss
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()
        train_progress_bar.set_postfix(batch_loss=loss.item())

      self.train_losses.append(train_loss / num_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss / num_images:.4f}")

      self.model.eval()
      val_loss = 0
      num_val_batches = len(val_loader)
      num_val_images = len(val_loader.dataset)
      correct_classification = 0
      correct_type = 0
      val_progress_bar = tqdm(enumerate(val_loader), total=num_val_batches, desc=f"Validation {epoch + 1}")
      with torch.no_grad():
        for _, (images, labels) in val_progress_bar:
          images = images.to(self.device)
          labels['class'] = labels['class'].to(self.device)
          labels['type'] = labels['type'].to(self.device)

          classification_output, type_output = self.forward(images)
          classification_loss = self.classification_loss_fn(classification_output, labels['class'])
          type_loss = self.type_loss_fn(type_output, labels['type'])

          loss = classification_loss + type_loss
          val_loss += loss.item()
          val_progress_bar.set_postfix(batch_loss=loss.item())

          classification_predictions = torch.argmax(classification_output, dim=1)
          type_predictions = torch.argmax(type_output, dim=1)

          correct_classification += (classification_predictions == labels['class']).sum().item()
          correct_type += (type_predictions == labels['type']).sum().item()

      classification_accuracy = correct_classification / num_val_images
      type_accuracy = correct_type / num_val_images

      self.val_losses.append(val_loss / num_val_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss / num_val_images:.4f}")

      print(f"Classification Accuracy: {classification_accuracy:.4f}")
      print(f"Type Accuracy: {type_accuracy:.4f}")

      self.classification_accuracies.append(classification_accuracy)
      self.type_accuracies.append(type_accuracy)

  def show_learning_curves(self, classification_accuracies, type_accuracies):
    if self.epochs <= 0:
      raise ValueError(f"Invalid epochs {self.epochs}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot learning curves
    axes[0].plot(range(1, self.epochs + 1), self.train_losses, label="Training Loss", marker="o")
    axes[0].plot(range(1, self.epochs + 1), self.val_losses, label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Plot classification and type accuracy
    axes[1].plot(range(1, self.epochs + 1), classification_accuracies, label="Classification Accuracy", marker="o")
    axes[1].plot(range(1, self.epochs + 1), type_accuracies, label="Type Accuracy", marker="o")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
