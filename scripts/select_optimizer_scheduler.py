import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torchvision.models as models
from resnet.resnet import ResNet
from classification.dataset import ENTRepDataset
from classification.transform import get_transform
from classification.evaluate import evaluate_model
from classification.make_submission import make_submission
from utils.data import *
from sklearn.model_selection import train_test_split
import pickle
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def save_to_disk(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Select optimizer and scheduler settings from a JSON file.')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file for optimizer/scheduler selection space.')
    return parser.parse_args()

def parse_accuracy_from_report(report_path):
    with open(report_path, 'r') as f:
        for line in f:
            if line.strip().startswith('accuracy'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
    return 0.0

def train(
    df,
    label_encoder,
    train_loader,
    val_loader,
    public_loader,
    optimizer,
    optimizer_name,
    optimizer_kwargs,
    scheduler,
    scheduler_name,
    scheduler_kwargs
):
    """
    Trains a ResNet model with the given optimizer and scheduler, evaluates on validation and public sets,
    and returns the accuracy scores.

    Args:
        df (pd.DataFrame): Training dataframe.
        label_encoder (dict): Mapping from class names to integer labels.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        public_loader (DataLoader): DataLoader for public test data.
        optimizer (type): Optimizer class.
        optimizer_name (str): Name of the optimizer.
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        scheduler (type): Scheduler class.
        scheduler_name (str): Name of the scheduler.
        scheduler_kwargs (dict): Keyword arguments for the scheduler.

    Returns:
        tuple: (validation accuracy, public accuracy)
    """
    # Initialize model
    model = ResNet(
        backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        hidden_channel=512,
        earlyStopping_patience=10,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
    )

    # Train model
    model.fine_tune(
        train_loader,
        val_loader,
        epochs=100,
        unfreeze_layers=['fc', 'layer4']
    )

    # Show learning curves
    learning_curve_path = f'./results/resnet50_{optimizer_name}_{scheduler_name}_hold_out_learning_curve.png'
    model.show_learning_curves(learning_curve_path)

    # Save model state
    model_path = f'ResNet50_{optimizer_name}_{scheduler_name}.pth'
    model.save_model_state(model_path)

    # Reload model for evaluation
    saved_model = ResNet.load_model(
        model_path,
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        512
    )

    # Define result file paths
    val_conf_matrix_path = f'./results/val_df_{optimizer_name}_{scheduler_name}_confusion_matrix.png'
    val_report_path = f'./results/val_df_{optimizer_name}_{scheduler_name}_classification_report.txt'
    public_conf_matrix_path = f'./results/public_df_{optimizer_name}_{scheduler_name}_confusion_matrix.png'
    public_report_path = f'./results/public_df_{optimizer_name}_{scheduler_name}_classification_report.txt'

    # Evaluate on validation set
    evaluate_model(
        saved_model,
        val_loader,
        label_encoder,
        val_conf_matrix_path,
        val_report_path
    )

    # Evaluate on public set
    evaluate_model(
        saved_model,
        public_loader,
        label_encoder,
        public_conf_matrix_path,
        public_report_path
    )

    # Parse accuracy from reports
    val_acc = parse_accuracy_from_report(val_report_path)
    public_acc = parse_accuracy_from_report(public_report_path)

    return val_acc, public_acc


def main():
    args = parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        selection_space = [
            config['selection_space']['schedulers'],
            config['selection_space']['optimizers']
        ]
        optimizer_map = {k: eval(v) for k, v in config['optimizer_map'].items()}
        scheduler_map = {k: eval(v) for k, v in config['scheduler_map'].items()}
        optimizer_kwargs_space = config['optimizer_kwargs_space']
        scheduler_kwargs_space = config['scheduler_kwargs_space']
    else:
        raise ValueError("\nNo found JSON config file for optimizer/scheduler selection space.")

    df = get_classification_task_train_df()
    test_df = get_classification_task_test_df()
    public_df = get_public_df()
    public_df['Label'] = public_df['Classification']

    label_encoder = {
        "nose-right": 0, 
        "nose-left" : 1, 
        "ear-right" : 2, 
        "ear-left"  : 3, 
        "vc-open"   : 4, 
        "vc-closed" : 5, 
        "throat"    : 6, 
    }

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    train_dataset = ENTRepDataset(
        train_df,
        label_encoder,
        transform=get_transform(train=True),
        is_train = True
    )
    val_dataset = ENTRepDataset(
        val_df,
        label_encoder,
        transform=get_transform(train=False)
    )
    public_dataset = ENTRepDataset(
       public_df, 
       label_encoder, 
       transform=get_transform(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
    public_loader = DataLoader(public_dataset, batch_size=4, shuffle=False, pin_memory=True)

    # Save datasets and dataloaders
    save_to_disk(train_dataset, 'train_dataset.pkl')
    save_to_disk(val_dataset, 'val_dataset.pkl')
    save_to_disk(public_dataset, 'public_dataset.pkl')

    save_to_disk(train_loader, 'train_loader.pkl')
    save_to_disk(val_loader, 'val_loader.pkl')
    save_to_disk(public_loader, 'public_loader.pkl')
    
    val_acc_results = {opt: {sch: 0 for sch in selection_space[0]} for opt in selection_space[1]}
    public_acc_results = {opt: {sch: 0 for sch in selection_space[0]} for opt in selection_space[1]}
    for scheduler_name in selection_space[0]:
        for optimizer_name in selection_space[1]:
            print("="* 50)
            print(f"Training with optimizer: {optimizer_name}, scheduler: {scheduler_name}")
            print("="* 50)

            optimizer = optimizer_map[optimizer_name]
            scheduler = scheduler_map[scheduler_name]
            optimizer_kwargs = optimizer_kwargs_space[optimizer_name]
            scheduler_kwargs = scheduler_kwargs_space[scheduler_name]
            val_acc, public_acc = train(
                df,
                label_encoder,
                train_loader,
                val_loader,
                public_loader,
                optimizer,
                optimizer_name,
                optimizer_kwargs,
                scheduler,
                scheduler_name,
                scheduler_kwargs
            )
            val_acc_results[optimizer_name][scheduler_name] = val_acc
            public_acc_results[optimizer_name][scheduler_name] = public_acc

            print("="* 50, "\n")

    # Plot heatmaps
    val_acc_df = pd.DataFrame(val_acc_results).T[selection_space[0]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_acc_df, annot=True, cmap='Blues')
    plt.title('Validation Accuracy Heatmap')
    plt.xlabel('Scheduler')
    plt.ylabel('Optimizer')
    plt.savefig('val_accuracy_heatmap.png')
    plt.close()

    public_acc_df = pd.DataFrame(public_acc_results).T[selection_space[0]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(public_acc_df, annot=True, cmap='Blues')
    plt.title('Test Accuracy Heatmap')
    plt.xlabel('Scheduler')
    plt.ylabel('Optimizer')
    plt.savefig('public_accuracy_heatmap.png')
    plt.close()

if __name__ == '__main__':
  main()