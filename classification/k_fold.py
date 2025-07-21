from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import DataLoader
from classification.dataset import ENTRepDataset
from classification.transform import get_transform
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

class K_Fold:
  def __init__(
      self, 
      k:int, 
      df: pd.DataFrame, 
      model, 
      label_encoder:dict,
      epochs: int,
      unfreeze_layers:list[str],
      batch_size:int = 4,
      random_seed: int = 42
    ):
    self.k = k
    self.skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    self.df = df
    self.fold_results = []
    self.best_model_state_dict = None
    self.best_accuracy = 0
    self.model = model
    self.label_encoder = label_encoder
    self.epochs = epochs
    self.unfreeze_layers = unfreeze_layers
    self.batch_size = batch_size
    self.train_losses = []
    self.val_losses = []
    self.accuracies = []
    self.random_seed = random_seed

  def run(self):
    """Run k-fold cross validation training"""
    print(f"Starting {self.k}-fold cross validation training...")
    
    for fold, (train_idx, val_idx) in enumerate(self.skf.split(self.df, self.df['Classification'])):
      print(f"\n{'='*50}")
      print(f"Fold {fold + 1}/{self.k}")
      print(f"{'='*50}")
      
      train_df = self.df.iloc[train_idx]
      val_df = self.df.iloc[val_idx]

      print(f"Training samples: {len(train_df)}")
      print(f"Validation samples: {len(val_df)}")

      train_dataset = ENTRepDataset(
         train_df, 
         self.label_encoder,
         transform=get_transform(train=True),
         is_train = True
      )
      val_dataset = ENTRepDataset(
         val_df, 
         self.label_encoder,
         transform=get_transform(train=False)
      )

      train_loader = DataLoader(
          train_dataset, 
          batch_size=self.batch_size, 
          shuffle=True,
          pin_memory=True
      )
      val_loader = DataLoader(
          val_dataset, 
          batch_size=self.batch_size, 
          shuffle=False,
          pin_memory=True
      )

      fold_model = deepcopy(self.model)

      # Train the model for this fold
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

      # Store learning curves for this fold
      max_epochs = self.epochs
      self.train_losses.append(fold_model.train_losses + [None] * (max_epochs - len(fold_model.train_losses)))
      self.val_losses.append(fold_model.val_losses + [None] * (max_epochs - len(fold_model.val_losses)))
      self.accuracies.append(fold_model.classification_accuracies + [None] * (max_epochs - len(fold_model.classification_accuracies)))

    # Print final results
    print(f"\n{'='*50}")
    print("K-Fold Cross-Validation Results")
    print(f"{'='*50}")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in self.fold_results]}")
    print(f"Average accuracy: {sum(self.fold_results) / len(self.fold_results):.4f}")
    print(f"Best accuracy: {self.best_accuracy:.4f}")
    print(f"Standard deviation: {np.std(self.fold_results):.4f}")

  def get_best_model_state_dict(self):
    """Get the best model state dictionary from all folds"""
    if self.best_model_state_dict is None:
      raise ValueError("No model has been trained yet. Call run() first.")
    return self.best_model_state_dict
  
  def get_fold_results(self):
    """Get the results from all folds"""
    return {
        'fold_accuracies': self.fold_results,
        'average_accuracy': sum(self.fold_results) / len(self.fold_results),
        'best_accuracy': self.best_accuracy,
        'std_accuracy': np.std(self.fold_results)
    }
  
  def show_learning_curves(self, save_path=None):
    """Plot learning curves for all folds"""
    if not self.fold_results:
        raise ValueError("No fold results available to plot. Call run() first.")

    num_folds = self.k
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot training losses for each fold
    for fold_idx in range(num_folds):
        valid_train_losses = [loss for loss in self.train_losses[fold_idx] if loss is not None]
        if valid_train_losses:
            axes[0, 0].plot(range(1, len(valid_train_losses) + 1), valid_train_losses, 
                           label=f"Fold {fold_idx + 1}", alpha=0.7)
    axes[0, 0].set_title("Training Loss Across Folds")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot validation losses for each fold
    for fold_idx in range(num_folds):
        valid_val_losses = [loss for loss in self.val_losses[fold_idx] if loss is not None]
        if valid_val_losses:
            axes[0, 1].plot(range(1, len(valid_val_losses) + 1), valid_val_losses, 
                           label=f"Fold {fold_idx + 1}", alpha=0.7)
    axes[0, 1].set_title("Validation Loss Across Folds")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot accuracies for each fold
    for fold_idx in range(num_folds):
        valid_accuracies = [accuracy for accuracy in self.accuracies[fold_idx] if accuracy is not None]
        if valid_accuracies:
            axes[1, 0].plot(range(1, len(valid_accuracies) + 1), valid_accuracies, 
                           label=f"Fold {fold_idx + 1}", alpha=0.7)
    axes[1, 0].set_title("Accuracy Across Folds")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot fold comparison
    fold_numbers = list(range(1, num_folds + 1))
    axes[1, 1].bar(fold_numbers, self.fold_results, alpha=0.7)
    axes[1, 1].axhline(y=sum(self.fold_results) / len(self.fold_results), 
                       color='red', linestyle='--', label='Average')
    axes[1, 1].set_title("Fold Performance Comparison")
    axes[1, 1].set_xlabel("Fold Number")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add text with summary statistics
    avg_acc = sum(self.fold_results) / len(self.fold_results)
    std_acc = np.std(self.fold_results)
    fig.suptitle(f'K-Fold Cross-Validation Results (k={self.k})\n'
                f'Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}', 
                fontsize=14, y=0.98)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve figure saved to {save_path}")
    plt.show()