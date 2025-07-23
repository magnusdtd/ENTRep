from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils.early_stopping import EarlyStopping

class Trainer:
    def __init__(
            self, 
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            scheduler=None,
            save_path: str="./results/clip_best.pt", 
            earlyStopping_patience: int = 7,
        ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        self.epochs = 0
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.early_stopping = EarlyStopping(
            patience=earlyStopping_patience,
            metric_name='Val Loss', 
            mode='min'
        )

    def train(self, epochs: int):
        self.epochs = epochs
        self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            num_batches = len(self.train_loader)
            num_samples = len(self.train_loader.dataset)
            train_progress_bar = tqdm(enumerate(self.train_loader), total=num_batches, desc=f"Epoch {epoch} [Train]")
            for _, batch in train_progress_bar:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                self.optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch["image"].size(0)
                train_progress_bar.set_postfix(batch_loss=loss.item())
            train_loss /= num_samples
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            num_batches = len(self.val_loader)
            num_samples = len(self.val_loader.dataset)
            val_progress_bar = tqdm(enumerate(self.val_loader), total=num_batches, desc=f"Epoch {epoch} [Validation]")
            with torch.no_grad():
                for _, batch in val_progress_bar:
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(self.device)
                    loss = self.model(batch)
                    val_loss += loss.item() * batch["image"].size(0)
                    val_progress_bar.set_postfix(batch_loss=loss.item())
            val_loss /= num_samples
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            if val_loss <= min(self.val_losses):
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Best model saved at epoch {epoch} with val loss {val_loss:.4f}")

            # Early stopping logic
            if self.early_stopping is not None:
                self.early_stopping(self.model, val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    # Load the best model state before breaking
                    if self.early_stopping.best_model_state is not None:
                        self.model.load_state_dict(self.early_stopping.best_model_state)
                    self.epochs = epoch + 1
                    break
                
    def show_learning_curves(self, save_path: str = None):
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            print("No training history to plot.")
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
        ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()
