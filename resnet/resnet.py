import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.earlyStopping import EarlyStopping
from utils.unfreeze_layer import unfreeze_model_layers
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ResNet:
  def __init__(
      self, 
      backbone, 
      lr:int=1e-3,
      num_classes: int = 7
):
    self.num_classes = num_classes

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model = backbone.to(self.device)
    self.model.fc = nn.Sequential(
      nn.Linear(self.model.fc.in_features, 256),
      nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(256, self.num_classes)
    ).to(self.device)

    self.classification_loss_fn = nn.CrossEntropyLoss().to(self.device)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
    self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
    self.epochs = 0

    self.train_losses = []
    self.val_losses = []
    self.classification_accuracies = []

    self.earlyStopping = EarlyStopping(patience=7)

  def forward(self, images: torch.Tensor):
    images = images.to(self.device)
    outputs = self.model(images)
    return outputs

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
        labels = labels['class']
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.forward(images)
        classification_loss = self.classification_loss_fn(outputs, labels)

        classification_loss.backward()
        self.optimizer.step()

        train_loss += classification_loss.item()
        train_progress_bar.set_postfix(batch_loss=classification_loss.item())

      self.train_losses.append(train_loss / num_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss / num_images:.4f}")

      self.model.eval()
      val_loss = 0
      num_val_batches = len(val_loader)
      num_val_images = len(val_loader.dataset)
      correct_classification = 0
      val_progress_bar = tqdm(enumerate(val_loader), total=num_val_batches, desc=f"Validation {epoch + 1}")
      with torch.no_grad():
        for _, (images, labels) in val_progress_bar:
          images = images.to(self.device)
          labels = labels['class']
          labels = labels.to(self.device)

          outputs = self.forward(images)
          classification_loss = self.classification_loss_fn(outputs, labels)

          val_loss += classification_loss.item()
          val_progress_bar.set_postfix(batch_loss=classification_loss.item())

          classification_predictions = torch.argmax(outputs, dim=1)
          correct_classification += (classification_predictions == labels).sum().item()

      classification_accuracy = correct_classification / num_val_images

      self.val_losses.append(val_loss / num_val_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss / num_val_images:.4f}")

      print(f"Classification Accuracy: {classification_accuracy:.4f}")

      self.classification_accuracies.append(classification_accuracy)

      self.scheduler.step(val_loss / num_val_images)

  def show_learning_curves(self):
    if self.epochs <= 0:
      raise ValueError(f"Invalid epochs {self.epochs}")

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot learning curves
    axes[0].plot(range(1, self.epochs + 1), self.train_losses, label="Training Loss", marker="o")
    axes[0].plot(range(1, self.epochs + 1), self.val_losses, label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Plot classification accuracy
    axes[1].plot(range(1, self.epochs + 1), self.classification_accuracies, label="Classification Accuracy", marker="o")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

  def fine_tune(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, unfreeze_layers: list = None):
    """
    Fine-tune the model with early stopping and layer unfreezing.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of epochs for fine-tuning.
        unfreeze_layers (list): List of layer names to unfreeze.
    """
    if unfreeze_layers:
      unfreeze_model_layers(self.model, unfreeze_layers)

    self.epochs = epochs

    for epoch in range(self.epochs):
      self.model.train()
      train_loss = 0

      num_batches = len(train_loader)
      num_images = len(train_loader.dataset)
      train_progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch + 1}")

      for _, (images, labels) in train_progress_bar:
        images = images.to(self.device)
        labels = labels['class']
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.forward(images)
        classification_loss = self.classification_loss_fn(outputs, labels)

        classification_loss.backward()
        self.optimizer.step()

        train_loss += classification_loss.item()
        train_progress_bar.set_postfix(batch_loss=classification_loss.item())

      self.train_losses.append(train_loss / num_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss / num_images:.4f}")

      self.model.eval()
      val_loss = 0
      num_val_batches = len(val_loader)
      num_val_images = len(val_loader.dataset)
      correct_classification = 0
      val_progress_bar = tqdm(enumerate(val_loader), total=num_val_batches, desc=f"Validation {epoch + 1}")
      with torch.no_grad():
        for _, (images, labels) in val_progress_bar:
          images = images.to(self.device)
          labels = labels['class']
          labels = labels.to(self.device)

          outputs = self.forward(images)
          classification_loss = self.classification_loss_fn(outputs, labels)

          val_loss += classification_loss.item()
          val_progress_bar.set_postfix(batch_loss=classification_loss.item())

          classification_predictions = torch.argmax(outputs, dim=1)
          correct_classification += (classification_predictions == labels).sum().item()

      classification_accuracy = correct_classification / num_val_images

      self.val_losses.append(val_loss / num_val_images)
      print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss / num_val_images:.4f}")

      print(f"Classification Accuracy: {classification_accuracy:.4f}")

      self.classification_accuracies.append(classification_accuracy)

      self.scheduler.step(val_loss / num_val_images)

      self.earlyStopping(self.model, classification_accuracy)
      if self.earlyStopping.early_stop:
        print("Early stopping triggered.")
        self.epochs = epoch + 1
        break

    # Ensure data lists match the actual number of completed epochs
    self.train_losses = self.train_losses[:self.epochs]
    self.val_losses = self.val_losses[:self.epochs]
    self.classification_accuracies = self.classification_accuracies[:self.epochs]

  def save_model_state(self, save_path: str):
    """Save the state dictionary of the model to the specified path."""
    torch.save(self.model.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")
