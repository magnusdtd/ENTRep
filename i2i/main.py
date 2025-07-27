import torch, os
from i2i.create_artifacts import create_artifacts
from i2i.artifact import  load_artifacts
from i2i.train import *
from i2i.ENTRepDataset import ENTRepDataset
from i2i.evaluate import evaluate_retrieval
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  exp_name = 'proto_clf_DINOv2s_BioCLIP_SAMViTB'
  if not os.path.exists(f'results/train_df_{exp_name}.csv'):
    train_df, val_df, test_df = create_artifacts(exp_name)
  else:
    train_df, val_df, test_df = load_artifacts(exp_name)

  def get_transform(image_size: Tuple[int, int] = (480, 640)):
      return A.Compose([
          A.Resize(*image_size),
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2()  
      ])

  # Use pair_mode=True for train/val
  train_dataset = ENTRepDataset(train_df, transform=get_transform(), pair_mode=True)
  val_dataset = ENTRepDataset(val_df, transform=get_transform(), pair_mode=True)
  train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
  )
  val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=32
  )
  input_dim = len(train_df['query_embedding'].iloc[0])
  projection_dim = 512

  model = ProjectionModel(
      input_dim=input_dim,
      embedder_dims=[input_dim],
      projection_dim=projection_dim,
      use_layernorm=False,
      use_dropout=True,
      dropout_rate=0.1,
      use_attention=False,
      internal_dim=1024,
      extra_layer=False
  )

  fitted_model = train(
      model, 
      train_loader, 
      val_loader,
      num_epochs=40,
      patience=7,
      lr=1e-4,
      device=device
  )

  test_dataset = ENTRepDataset(test_df, transform=get_transform())
  test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=32
  )

  evaluate_retrieval(fitted_model, test_loader, device, top_k=5)

if __name__ == "__main__":
  main()
