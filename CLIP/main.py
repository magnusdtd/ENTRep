import torch
import argparse
from utils.data import *
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from CLIP.dataset import ENTRepDataset
from CLIP.transform import get_transform
from CLIP.CLIP import CLIP
from CLIP.trainer import Trainer
from CLIP.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Train CLIP model with configurable learning rate, epochs, and batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--recall_k', type=int, default=5, help='Value of k for recall@k evaluation')
    args = parser.parse_args()

    df = get_t2i_task_train_df()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,
        random_state=42
    )
    train_dataset = ENTRepDataset(
        train_df,
        tokenizer,
        max_length=64, 
        transform=get_transform(train=True),
    )
    val_dataset = ENTRepDataset(
        val_df,
        tokenizer,
        max_length=64, 
        transform=get_transform(train=False)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = CLIP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler=scheduler)
    trainer.train(epochs=args.epochs)
    trainer.show_learning_curves('./results/learning_curve.png')

    evaluator = Evaluator(model, val_loader)
    recall = evaluator.recall_at_k(k=args.recall_k)
    mrr = evaluator.mean_reciprocal_rank()
    with open('./results/eval_metrics.txt', 'w') as f:
        f.write(f"Recall@{args.recall_k}:\n")
        for k, v in recall.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nMRR:\n")
        for k, v in mrr.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()