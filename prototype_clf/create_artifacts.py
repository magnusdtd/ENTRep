import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from prototype_clf.artifact import make_artifacts, save_artifacts, load_artifacts
from prototype_clf.ENTRep import ENTRep
from prototype_clf.test_entrep import test_entrep_dataset


def create_artifacts():
  df = pd.read_json('Dataset/train/cls.json', orient='index')
  df = df.reset_index()
  df.columns = ['Path', 'Classification']
  df["Path"] = "Dataset/train/imgs/" + df["Path"].astype(str)

  class_feature_map = {
    "nose-right": 0, 
    "nose-left" : 1, 
    "ear-right" : 2, 
    "ear-left"  : 3, 
    "vc-open"   : 4, 
    "vc-closed" : 5, 
    "throat"    : 6, 
  }

  df['Classification'] = df['Classification'].map(class_feature_map)

  # Model configuration
  bioclip_args = {
    'model_name': 'hf-hub:magnusdtd/bio-clip-ft'
  }
  dinov2_args = {
    'repo_name': 'facebookresearch/dinov2', 
    'model_name': 'dinov2_vits14',
    'image_size': (490, 644)
  }
  samvit_args = {
    'model_name': 'samvit_base_patch16.sa1b'
  }

  print("Creating embeddings for training data...")
  feature_df = make_artifacts(df, bioclip_args, dinov2_args, samvit_args)

  # Split into train and validation sets
  train_df, val_df = train_test_split(feature_df, test_size=0.2, random_state=42, stratify=feature_df['Classification'])

  print("Loading test data...")
  test_df = pd.read_json('Dataset/public/data.json')
  test_df["Path"] = test_df["Path"].str.replace("_image", "_Image", regex=False)
  test_df["Path"] = "Dataset/public/images/" + test_df["Path"].astype(str)

  print("Creating embeddings for test data...")
  test_df = make_artifacts(test_df, bioclip_args, dinov2_args, samvit_args)

  # Verify all datasets have embeddings
  print(f"Train embeddings shape: {np.array(train_df['embedding'].tolist()).shape}")
  print(f"Validation embeddings shape: {np.array(val_df['embedding'].tolist()).shape}")
  print(f"Test embeddings shape: {np.array(test_df['embedding'].tolist()).shape}")

  config = {
      'bioclip_args': bioclip_args,
      'dinov2_args': dinov2_args,
      'samvit_args': samvit_args,
      'class_feature_map': class_feature_map,
      'embedding_dim': feature_df['emb_dims'].iloc[0]
  }

  train_dataset = ENTRep(train_df)
  val_dataset = ENTRep(val_df)
  test_dataset = ENTRep(test_df)

  print("Saving artifacts...")
  save_artifacts("proto_clf_DINOv2s_BioCLIP_SAMViTB", train_dataset, val_dataset, test_dataset, config)

  print("Validating artifacts by loading...")
  loaded_train_df, loaded_val_df, loaded_test_df = load_artifacts("proto_clf_DINOv2s_BioCLIP_SAMViTB")

  # Verify the loaded data
  print(f"Original train shape: {train_df.shape}")
  print(f"Loaded train shape: {loaded_train_df.shape}")
  print(f"Original val shape: {val_df.shape}")
  print(f"Loaded val shape: {loaded_val_df.shape}")
  print(f"Original test shape: {test_df.shape}")
  print(f"Loaded test shape: {loaded_test_df.shape}")

  # Check if embeddings are preserved
  train_embeddings_match = np.array_equal(
      np.array(train_df['embedding'].tolist()), 
      np.array(loaded_train_df['embedding'].tolist())
  )
  print(f"Train embeddings match: {train_embeddings_match}")

  val_embeddings_match = np.array_equal(
      np.array(val_df['embedding'].tolist()), 
      np.array(loaded_val_df['embedding'].tolist())
  )
  print(f"Validation embeddings match: {val_embeddings_match}")

  test_embeddings_match = np.array_equal(
      np.array(test_df['embedding'].tolist()), 
      np.array(loaded_test_df['embedding'].tolist())
  )
  print(f"Test embeddings match: {test_embeddings_match}")

  print("Artifact creation and validation completed successfully!")

  test_entrep_dataset()

  return loaded_train_df, loaded_val_df, loaded_test_df

