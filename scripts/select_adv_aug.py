import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torchvision.models as models
from resnet.resnet import ResNet
from classification.dataset import ENTRepDataset
from classification.transform import get_transform
from classification.evaluate import evaluate_model
from utils.data import *
from sklearn.model_selection import train_test_split
import pickle
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt

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
    scheduler_kwargs,
    use_mixup=False,
    use_cutmix=False,
    use_mosaic=False
):
    # Initialize model
    model = ResNet(
        backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        hidden_channel=512,
        earlyStopping_patience=10,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        use_mixup=use_mixup,
        use_cutmix=use_cutmix,
        use_mosaic=use_mosaic
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
        optimizer_name = config['optimizer']
        scheduler_name = config['scheduler_map']
        optimizer = eval(optimizer_name)
        scheduler = eval(scheduler_name)
        optimizer_kwargs = config['optimizer_kwargs_space']
        scheduler_kwargs = config['scheduler_kwargs_space']
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
    
    # Define augmentation modes
    aug_modes = [
        {"name": "No advanced augmentation", "mixup": False, "cutmix": False, "mosaic": False},
        {"name": "CutMix", "mixup": False, "cutmix": True, "mosaic": False},
        {"name": "MixUp", "mixup": True, "cutmix": False, "mosaic": False},
        {"name": "Mosaic", "mixup": False, "cutmix": False, "mosaic": True},
        {"name": "CutMix + MixUp", "mixup": True, "cutmix": True, "mosaic": False},
        {"name": "CutMix + Mosaic", "mixup": False, "cutmix": True, "mosaic": True},
        {"name": "MixUp + Mosaic", "mixup": True, "cutmix": False, "mosaic": True},
        {"name": "CutMix + MixUp + Mosaic", "mixup": True, "cutmix": True, "mosaic": True},
    ]

    val_acc_results = {}
    public_acc_results = {}

    for mode in aug_modes:
        print("="* 100)
        print(f"Training with advanced augmentations: {mode['name']}")
        print("="* 100)
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
            scheduler_kwargs,
            use_mixup=mode["mixup"],
            use_cutmix=mode["cutmix"],
            use_mosaic=mode["mosaic"]
        )
        val_acc_results[mode["name"]] = val_acc
        public_acc_results[mode["name"]] = public_acc
        print("="* 100, "\n")

    # Plotting
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(val_acc_results.keys()), y=list(val_acc_results.values()))
    plt.title('Validation Accuracy by Augmentation Mode')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Augmentation Mode')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('val_acc_by_aug_mode.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(public_acc_results.keys()), y=list(public_acc_results.values()))
    plt.title('Public Accuracy by Augmentation Mode')
    plt.ylabel('Public Accuracy')
    plt.xlabel('Augmentation Mode')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('public_acc_by_aug_mode.png')
    plt.show()


if __name__ == '__main__':
  main()