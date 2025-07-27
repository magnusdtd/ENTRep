import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torchvision.models as models
from SwinTransformer.swin_transformer import SwinTransformer
from classification.dataset import ENTRepDataset
from classification.transform import get_transform
from classification.evaluate import evaluate_model
from classification.make_submission import make_submission
from classification.k_fold import K_Fold
from utils.data import *
import argparse
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def parse_args():
    parser = argparse.ArgumentParser(description='SwinTransformer Tiny Training/Evaluation Script')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--k', type=int, default=5, help='The value of k for Recall@k and MRR.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--mode', type=str, choices=['hold-out', 'k-fold'], default='hold-out', help='Training mode: hold-out or k-fold.')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for k-fold cross-validation.')
    return parser.parse_args()

def main(args):
    df = get_classification_task_train_df()
    label_encoder = {
        "nose-right": 0, 
        "nose-left" : 1, 
        "ear-right" : 2, 
        "ear-left"  : 3, 
        "vc-open"   : 4, 
        "vc-closed" : 5, 
        "throat"    : 6, 
    }

    model = SwinTransformer(
      backbone=models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
      hidden_channel=512,
      earlyStopping_patience=10,
      optimizer_kwargs={
          'lr': args.lr,
          'weight_decay': 0
      },
      scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
      scheduler_kwargs={
          'T_max': 100,
          'eta_min': 1e-7
      },
      use_mixup=True,
      use_cutmix=True,
      use_mosaic=True
    )

    if args.mode == 'hold-out':
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

        train_dataset = ENTRepDataset(
            train_df,
            label_encoder,
            transform=get_transform(train=True),
            is_train=True
        )
        val_dataset = ENTRepDataset(
            val_df,
            label_encoder,
            transform=get_transform(train=False)
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=4, 
            shuffle=False,
            pin_memory=True
        )

        model.fine_tune(
            train_loader,
            val_loader,
            epochs=args.epochs,
            unfreeze_layers=[
                'head', 
                'norm',
                'features.7.1',
            ]
        )

        model.show_learning_curves('./results/swint_hold_out_learning_curve.png')

    elif args.mode == 'k-fold':
        kf = K_Fold(
            k=args.folds, 
            df=df, 
            model=model, 
            label_encoder=label_encoder,
            epochs=args.epochs,
            unfreeze_layers=[
                'head', 
                'norm',
                'features.7.1',
            ]
        )
        kf.run()
        kf.show_learning_curves('./results/swint_k_fold_learning_curve.png')
        model.load_state_dict(kf.get_best_model_state_dict())

    exp_name = 'Swin_T.pth'
    model.save_model_state(exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = SwinTransformer.load_model(
        exp_name, 
        models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
        512
    )

    # Model evaluation
    dataset = ENTRepDataset(df, label_encoder, transform=get_transform(train=False))
    dataLoader = DataLoader(dataset, batch_size=4, shuffle=True)

    evaluate_model(
        saved_model, 
        dataLoader, 
        label_encoder, 
        './results/train_df_confusion_matrix.png', 
        './results/train_df_classification_report.txt'
    )

    # Make submission
    test_df = get_classification_task_test_df()
    make_submission(saved_model, 'Swin_T', device, test_df)

    # Evaluate with public set
    public_df = get_public_df()
    public_df['Label'] = public_df['Classification']
    public_dataset = ENTRepDataset(public_df, label_encoder, transform=get_transform(train=False))
    public_dataLoader = DataLoader(public_dataset, batch_size=4, shuffle=True)
    evaluate_model(
        saved_model, 
        public_dataLoader, 
        label_encoder, 
        './results/public_df_confusion_matrix.png', 
        './results/public_df_classification_report.txt'
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
