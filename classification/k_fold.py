from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import DataLoader
from classification.dataset import ENTRepDataset
from classification.transform import get_transform
import matplotlib.pyplot as plt
from copy import deepcopy

class K_Fold:
  def __init__(
      self, 
      k:int, 
      df: pd.DataFrame, 
      model, 
      class_feature_map:dict,
      epochs: int,
      unfreeze_layers:list[str],
      batch_size:int = 4,
      img_path = 'Dataset/train/imgs'
    ):
    self.k = k
    self.skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    self.df = df
    self.fold_results = []
    self.best_model_state_dict = None
    self.best_accuracy = 0
    self.model = model
    self.class_feature_map = class_feature_map
    self.epochs = epochs
    self.unfreeze_layers = unfreeze_layers
    self.batch_size = batch_size
    self.train_losses = []
    self.val_losses = []
    self.accuracies = []
    self.img_path = img_path

  def run(self):
    for fold, (train_idx, val_idx) in enumerate(self.skf.split(self.df, self.df['Classification'])):
      print(f"Fold {fold + 1}/{self.k}")
      
      train_df = self.df.iloc[train_idx]
      val_df = self.df.iloc[val_idx]

      train_dataset = ENTRepDataset(
         train_df, 
         self.class_feature_map, 
         img_path=self.img_path, 
         transform=get_transform(train=True)
      )
      val_dataset = ENTRepDataset(
         val_df, 
         self.class_feature_map, 
         img_path=self.img_path, 
         transform=get_transform(train=False)
      )

      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

      fold_model = deepcopy(self.model)

      fold_accuracy = fold_model.fine_tune(
        train_loader,
        val_loader,
        epochs=self.epochs,
        unfreeze_layers=self.unfreeze_layers
      )

      if fold_accuracy > self.best_accuracy:
        self.best_accuracy = fold_accuracy
        self.best_model_state_dict = fold_model.earlyStopping.best_model_state

      self.fold_results.append(fold_accuracy)

      # Ensure train_losses, val_losses, and accuracies are padded to match the number of epochs
      max_epochs = self.epochs
      self.train_losses.append(fold_model.train_losses + [None] * (max_epochs - len(fold_model.train_losses)))
      self.val_losses.append(fold_model.val_losses + [None] * (max_epochs - len(fold_model.val_losses)))
      self.accuracies.append(fold_model.classification_accuracies + [None] * (max_epochs - len(fold_model.classification_accuracies)))

    print("K-Fold Cross-Validation Results:", self.fold_results)
    print("Average Accuracy:", sum(self.fold_results) / len(self.fold_results))

  def get_best_model_state_dict(self):
    return self.best_model_state_dict
  
  # Handle None values in plotting by skipping them
  def show_learning_curves(self):
    if not self.fold_results:
        raise ValueError("No fold results available to plot.")

    num_folds = self.k
    _, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot training losses for each fold
    for fold_idx in range(num_folds):
        valid_train_losses = [loss for loss in self.train_losses[fold_idx] if loss is not None]
        axes[0, 0].plot(range(1, len(valid_train_losses) + 1), valid_train_losses, label=f"Fold {fold_idx + 1}", marker="o")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot validation losses for each fold
    for fold_idx in range(num_folds):
        valid_val_losses = [loss for loss in self.val_losses[fold_idx] if loss is not None]
        axes[0, 1].plot(range(1, len(valid_val_losses) + 1), valid_val_losses, label=f"Fold {fold_idx + 1}", marker="o")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot accuracies for each fold
    for fold_idx in range(num_folds):
        valid_accuracies = [accuracy for accuracy in self.accuracies[fold_idx] if accuracy is not None]
        axes[1, 0].plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label=f"Fold {fold_idx + 1}", marker="o")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()