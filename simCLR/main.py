from utils.data import *
from simCLR.dataset import ENTRepDataset
from simCLR.model import SimCLR
from simCLR.transform import get_transform
from simCLR.trainer import SimCLRTrainer
from simCLR.evaluator import SimCLREvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
import random
import numpy as np
import torch
import argparse
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def save_to_disk(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_from_disk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Training/Evaluation Script')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs.')
    parser.add_argument('--k', type=int, default=5, help='The value of k for Recall@k and MRR.')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for training.')
    return parser.parse_args()

def main(args):
    data_dir = './data'
    train_pkl = os.path.join(data_dir, 'train_dataset.pkl')
    val_pkl = os.path.join(data_dir, 'val_dataset.pkl')

    if os.path.exists(train_pkl) and os.path.exists(val_pkl):
        print('Loading datasets from pickle files...')
        train_dataset = load_from_disk(train_pkl)
        val_dataset = load_from_disk(val_pkl)
    else:
        print('Pickle files not found. Creating datasets from scratch...')
        df = get_i2i_task_train_df()

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = ENTRepDataset(
            train_df,
            transform=get_transform(train=True),
        )
        val_dataset = ENTRepDataset(
            val_df,
            transform=get_transform(train=False)
        )
        # Save datasets
        os.makedirs(data_dir, exist_ok=True)
        save_to_disk(train_dataset, train_pkl)
        save_to_disk(val_dataset, val_pkl)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

    model = SimCLR()
    trainer = SimCLRTrainer(model, args.k, args.lr, train_loader, val_loader)
    trainer.train(args.epochs)
    trainer.save_model_state("./results/clip.pth")
    evaluator = SimCLREvaluator(model, val_loader)
    recall_k = evaluator.get_recall_k(args.k)
    mrr = evaluator.get_mrr()
    metrics_path = "./results/metrics.txt"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"Recall@{args.k}: {recall_k}\n")
        f.write(f"MRR: {mrr}\n")
    print(f"Recall@{args.k} = {recall_k}")
    print(f"MRR = {mrr}")

    trainer.show_learning_curves("./results/learning_curve.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)