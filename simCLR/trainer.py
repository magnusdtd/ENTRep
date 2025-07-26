
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from utils.early_stopping import EarlyStopping
from simCLR.nt_xent import NTXentLoss

class SimCLRTrainer:
    def __init__(self, model, k: int, lr: float, train_loader: DataLoader, val_loader: DataLoader):
        self.lr = lr
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = NTXentLoss(temperature=0.5)
        self.early_stopping = EarlyStopping(patience=10, metric_name=f'Recall@{self.k}', mode='max')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 100, eta_min = 1e-6)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.recall_k_values = []
        self.mrr_values = []

    def _compute_recall_mrr_positive_pairs(self, embeddings):
        """
        Compute recall@k and MRR for positive pairs (Path1, Path2) in the dataset.
        Assumes embeddings are ordered as [img1_row0, img2_row0, img1_row1, img2_row1, ...]
        """
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        N = embeddings.size(0)
        k = self.k
        recall_at_k = 0
        reciprocal_ranks = []
        num_pairs = N // 2
        for pair_idx in range(num_pairs):
            i = pair_idx * 2
            j = i + 1
            # For i, positive is j
            for anchor, positive in [(i, j), (j, i)]:
                sim_scores = similarity_matrix[anchor].clone()
                sim_scores[anchor] = -float('inf')  # Exclude self
                sorted_indices = torch.argsort(sim_scores, descending=True)
                # Find where the positive pair is in the ranking
                rank = (sorted_indices == positive).nonzero(as_tuple=True)[0].item() + 1
                if rank <= k:
                    recall_at_k += 1
                reciprocal_ranks.append(1.0 / rank)
        recall_at_k = recall_at_k / N
        mrr = sum(reciprocal_ranks) / N
        return recall_at_k, mrr

    def train(self, epochs=10):
        self.epochs = epochs
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            num_batches = len(self.train_loader)
            num_images = len(self.train_loader.dataset)
            train_progress_bar = tqdm(enumerate(self.train_loader), total=num_batches, desc=f"Epoch {epoch + 1}")
            for _, (xi, xj) in train_progress_bar:
                xi, xj = xi.to(self.device), xj.to(self.device)
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
                train_loss += loss.item()
                train_progress_bar.set_postfix(batch_loss=loss.item())
            self.train_losses.append(train_loss / num_images)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss / num_images:.4f}")

            val_loss = 0
            num_val_batches = len(self.val_loader)
            num_val_images = len(self.val_loader.dataset)
            val_progress_bar = tqdm(enumerate(self.val_loader), total=num_val_batches, desc=f"Validation {epoch + 1}")
            val_embeddings = []
            self.model.eval()
            with torch.no_grad():
                for _, (xi, xj) in val_progress_bar:
                    xi, xj = xi.to(self.device), xj.to(self.device)
                    zi = self.model(xi)
                    zj = self.model(xj)
                    val_loss += self.criterion(zi, zj).item()
                    val_progress_bar.set_postfix(batch_loss=self.criterion(zi, zj).item())
                    val_embeddings.append(zi.detach().cpu())
                    val_embeddings.append(zj.detach().cpu())
            self.val_losses.append(val_loss / num_val_images)

            # Calculate recall@k and MRR for positive pairs
            val_embeddings_tensor = torch.cat(val_embeddings, dim=0)
            recall_k, mrr = self._compute_recall_mrr_positive_pairs(val_embeddings_tensor)
            self.recall_k_values.append(recall_k)                
            self.mrr_values.append(mrr)
            print(f"Recall@{self.k}: {recall_k:.4f} | MRR: {mrr:.4f}")

            self.earlyStopping(self.model, recall_k)
            if self.earlyStopping.early_stop:
              print("Early stopping triggered.")
              # Load the best model state before breaking
              if self.earlyStopping.best_model_state is not None:
                self.model.load_state_dict(self.earlyStopping.best_model_state)
              self.epochs = epoch + 1
              break
    
    def save_model_state(self, save_path: str):
        """Save the state dictionary of the model to the specified path."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model state dictionary saved to {save_path}")


    def show_learning_curves(self, save_path: str = None):
        if len(self.train_losses) <= 0:
            raise ValueError(f"No training data available")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot learning curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, label="Training Loss")
        if self.val_losses:
            axes[0].plot(epochs, self.val_losses, label="Validation Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Learning Curve")
        axes[0].legend()
        axes[0].grid(True)

        # Plot recall@k
        if self.recall_k_values:
            axes[1].plot(epochs, self.recall_k_values, label="Recall@1", color='green')
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("Recall@1")
            axes[1].set_title("Recall@1 Curve")
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No validation data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("Recall@1 Curve")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()