import torch
import argparse
from utils.data import *
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json

from CLIP.dataset import ENTRepDataset
from CLIP.transform import get_transform
from CLIP.CLIP import CLIP
from CLIP.trainer import Trainer
from CLIP.evaluator import Evaluator

import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def main():
    # First, parse --config if present
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with arguments')
    config_args, remaining_argv = config_parser.parse_known_args()
    defaults = {}
    if config_args.config:
        with open(config_args.config, 'r') as f:
            config_defaults = json.load(f)
        defaults.update({str(k): v for k, v in config_defaults.items()})

    parser = argparse.ArgumentParser(
        description="Train CLIP model with configurable learning rate, epochs, batch size, optimizer, scheduler, and encoder layer unfreezing.",
        parents=[config_parser]
    )
    if isinstance(defaults.get("optimizer_kwargs"), dict):
        defaults["optimizer_kwargs"] = json.dumps(defaults["optimizer_kwargs"])
    if isinstance(defaults.get("scheduler_kwargs"), dict):
        defaults["scheduler_kwargs"] = json.dumps(defaults["scheduler_kwargs"])
    parser.set_defaults(**defaults)
    parser.add_argument('--lr', type=float, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for DataLoader')
    parser.add_argument('--recall_k', type=int, help='Value of k for recall@k evaluation')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW', 'SGD'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, choices=['reduce_on_plateau', 'cosine', 'step'], help='Scheduler type')
    parser.add_argument('--optimizer_kwargs', type=str, help='Additional optimizer kwargs as JSON string')
    parser.add_argument('--scheduler_kwargs', type=str, help='Additional scheduler kwargs as JSON string')
    parser.add_argument('--img_encoder_unfreeze_layers', type=str, help='Comma-separated list of image encoder layers to unfreeze')
    parser.add_argument('--text_encoder_unfreeze_layers', type=str, help='Comma-separated list of text encoder layers to unfreeze')
    args = parser.parse_args(remaining_argv)

    OPTIMIZERS = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD,
    }
    SCHEDULERS = {
        'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step': torch.optim.lr_scheduler.StepLR,
    }

    optimizer_kwargs = {'lr': args.lr}
    if args.optimizer == 'sgd':
        optimizer_kwargs.setdefault('momentum', 0.9)
    if args.optimizer_kwargs:
        optimizer_kwargs.update(json.loads(args.optimizer_kwargs))

    scheduler_kwargs = {}
    if args.scheduler == 'reduce_on_plateau':
        scheduler_kwargs.update({'mode': 'min', 'patience': 3})
    elif args.scheduler == 'cosine':
        scheduler_kwargs.update({'T_max': args.epochs, 'eta_min': 1e-6})
    elif args.scheduler == 'step':
        scheduler_kwargs.update({'step_size': 3, 'gamma': 0.1})
    if args.scheduler_kwargs:
        scheduler_kwargs.update(json.loads(args.scheduler_kwargs))

    img_unfreeze_layers = [l.strip() for l in args.img_encoder_unfreeze_layers.split(',') if l]
    text_unfreeze_layers = [l.strip() for l in args.text_encoder_unfreeze_layers.split(',') if l]

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
    model = CLIP(
        img_encoder_unfreeze_layers=img_unfreeze_layers,
        text_encoder_unfreeze_layers=text_unfreeze_layers
    )
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), **optimizer_kwargs)
    scheduler = SCHEDULERS[args.scheduler](optimizer, **scheduler_kwargs)
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