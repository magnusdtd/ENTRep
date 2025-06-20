import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.early_stopping import EarlyStopping
from utils.unfreeze_layer import unfreeze_model_layers
from utils.mixup import mixup_data, cutmix_data
import numpy as np


class Classification:
  def __init__(
      self,
      num_classes: int = 7,
      earlyStopping_patience: int = 7
):
    self.num_classes = num_classes

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.epochs = 0

    self.train_losses = []
    self.val_losses = []
    self.classification_accuracies = []

    self.earlyStopping = EarlyStopping(patience=earlyStopping_patience)

    self.model = None
    self.classification_loss_fn = None
    self.optimizer = None
    self.scheduler = None

  def forward(self, images: torch.Tensor):
    images = images.to(self.device)
    outputs = self.model(images)
    return outputs

  def show_learning_curves(self):
    if self.epochs <= 0:
      raise ValueError(f"Invalid epochs {self.epochs}")

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot learning curves
    axes[0].plot(range(1, self.epochs + 1), self.train_losses, label="Training Loss")
    axes[0].plot(range(1, self.epochs + 1), self.val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Plot classification accuracy
    axes[1].plot(range(1, self.epochs + 1), self.classification_accuracies, label="Classification Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

  def fine_tune(
      self, 
      train_loader: DataLoader, 
      val_loader: DataLoader, 
      epochs: int = 10, 
      unfreeze_layers: list = None, 
      use_mixup: bool = False, 
      mixup_alpha: float = 0.4, 
      use_cutmix: bool = False, 
      cutmix_alpha: float = 1.0
  ):
    """
    Fine-tune the model with early stopping and layer unfreezing.
    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of epochs for fine-tuning.
        unfreeze_layers (list): List of layer names to unfreeze.
        use_mixup (bool): Use MixUp augmentation if True.
        mixup_alpha (float): MixUp alpha parameter.
        use_cutmix (bool): Use CutMix augmentation if True.
        cutmix_alpha (float): CutMix alpha parameter.
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
        labels_class = labels['class'].to(self.device)

        self.optimizer.zero_grad()

        # ugmentation
        if use_mixup and use_cutmix:
          if np.random.rand() < 0.5:
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels_class, alpha=mixup_alpha)
          else:
            mixed_images, targets_a, targets_b, lam = cutmix_data(images, labels_class, alpha=cutmix_alpha)
          outputs = self.forward(mixed_images)
          loss = lam * self.classification_loss_fn(outputs, targets_a) + (1 - lam) * self.classification_loss_fn(outputs, targets_b)
        elif use_mixup:
          mixed_images, targets_a, targets_b, lam = mixup_data(images, labels_class, alpha=mixup_alpha)
          outputs = self.forward(mixed_images)
          loss = lam * self.classification_loss_fn(outputs, targets_a) + (1 - lam) * self.classification_loss_fn(outputs, targets_b)
        elif use_cutmix:
          mixed_images, targets_a, targets_b, lam = cutmix_data(images, labels_class, alpha=cutmix_alpha)
          outputs = self.forward(mixed_images)
          loss = lam * self.classification_loss_fn(outputs, targets_a) + (1 - lam) * self.classification_loss_fn(outputs, targets_b)
        else:
          outputs = self.forward(images)
          loss = self.classification_loss_fn(outputs, labels_class)

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
        # Load the best model state before breaking
        if self.earlyStopping.best_model_state is not None:
          self.model.load_state_dict(self.earlyStopping.best_model_state)
        self.epochs = epoch + 1
        break

    # Ensure data lists match the actual number of completed epochs
    self.train_losses = self.train_losses[:self.epochs]
    self.val_losses = self.val_losses[:self.epochs]
    self.classification_accuracies = self.classification_accuracies[:self.epochs]

    return self.earlyStopping.best_value

  def save_model_state(self, save_path: str):
    """Save the state dictionary of the model to the specified path."""
    torch.save(self.model.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")

  @staticmethod
  def load_model(model_path:str, backbone):
    """Load the trained model state dictionary into the model architecture."""
    pass