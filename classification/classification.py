import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.early_stopping import EarlyStopping
from utils.unfreeze_layer import unfreeze_model_layers
from utils.mixup import mixup_data
from utils.cutmix import cutmix_data
from utils.mosaic import mosaic_data
import numpy as np


class Classification:
  def __init__(
      self,
      earlyStopping_patience: int = 7,
      criterion=None,
      optimizer=torch.optim.Adam,
      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
      use_mixup = False,
      mixup_alpha = 0.4,
      use_cutmix = False,
      cutmix_alpha = 1.0,
      use_mosaic = False,
      adv_aug_prob = 0.5,
      **kwargs
):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.epochs = 0

    self.train_losses = []
    self.val_losses = []
    self.classification_accuracies = []

    self.earlyStopping = EarlyStopping(patience=earlyStopping_patience)

    self.use_mixup = use_mixup 
    self.mixup_alpha = mixup_alpha
    self.use_cutmix = use_cutmix 
    self.cutmix_alpha = cutmix_alpha
    self.use_mosaic = use_mosaic
    self.adv_aug_prob = adv_aug_prob

    # Loss function
    if criterion is None:
        class_weights = kwargs.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=self.device).float()
        self.classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(self.device).float()
    else:
        self.classification_loss_fn = criterion.to(self.device).float()

    # Optimizer
    optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 1e-4}
    if optimizer == torch.optim.SGD:
        optimizer_kwargs.update({'momentum': 0.9})
    optimizer_kwargs.update(kwargs.get('optimizer_kwargs', {}))
    self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

    # Scheduler
    scheduler_kwargs = {}
    if scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler_kwargs.update({'mode': 'min', 'patience': 3})
    elif scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
        scheduler_kwargs.update({'T_max': kwargs.get('T_max', 50)})
        scheduler_kwargs.update({'eta_min': kwargs.get('eta_min', 1e-6)})
    scheduler_kwargs.update(kwargs.get('scheduler_kwargs', {}))

    self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)

    self.model.to(self.device).float()

  def forward(self, images: torch.Tensor):
    images = images.to(self.device)
    outputs = self.model(images)
    return outputs

  def show_learning_curves(self, save_path: str = None):
    if self.epochs <= 0:
      raise ValueError(f"Invalid epochs {self.epochs}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
    if save_path is not None:
      fig.savefig(save_path)
    plt.show()

  def _apply_augmentation(self, images, labels_class):
      enabled_augs = []
      batch_size = images.size(0)
      if self.use_mixup:
          enabled_augs.append('mixup')
      if self.use_cutmix:
          enabled_augs.append('cutmix')
      if self.use_mosaic and batch_size >= 4:
          enabled_augs.append('mosaic')
      if not enabled_augs:
          return images, labels_class, labels_class, 1.0
      aug = np.random.choice(enabled_augs)
      if aug == 'mixup':
          return mixup_data(images, labels_class, alpha=self.mixup_alpha)
      elif aug == 'cutmix':
          return cutmix_data(images, labels_class, alpha=self.cutmix_alpha)
      elif aug == 'mosaic':
          mosaic_imgs, mosaic_labels, lam_list = mosaic_data(images, labels_class)
          return mosaic_imgs, mosaic_labels, lam_list, 'mosaic'
      else:
          return images, labels_class, labels_class, 1.0

  def fine_tune(
      self, 
      train_loader: DataLoader, 
      val_loader: DataLoader, 
      epochs: int = 10, 
      unfreeze_layers: list = None,
  ):
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
        labels_class = labels['class'].to(self.device)

        self.optimizer.zero_grad()

        # Augmentation
        if np.random.random() < self.adv_aug_prob and (self.use_mixup or self.use_cutmix or self.use_mosaic):
            aug_result = self._apply_augmentation(images, labels_class)
            if len(aug_result) == 4 and aug_result[-1] == 'mosaic':
                mixed_images, mosaic_labels, lam_list, _ = aug_result
                outputs = self.forward(mixed_images)
                # For each image, compute the weighted sum of losses for the 4 labels
                loss = 0
                for k in range(4):
                    loss += lam_list[:, k] * self.classification_loss_fn(outputs, mosaic_labels[:, k])
                loss = loss.sum() / images.size(0)
            else:
                mixed_images, targets_a, targets_b, lam = aug_result
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

  def load_state_dict(self, model_state_dict):
    self.model.load_state_dict(model_state_dict)