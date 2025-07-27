import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from i2i.artifact import make_artifacts, save_artifacts, load_artifacts
from i2i.ENTRepDataset import ENTRepDataset
from i2i.test_entrep import test_entrep_dataset


def create_artifacts(exp_name: str):
    # Read i2i.json as pairs
    with open('Dataset/train/i2i.json', 'r') as f:
        pairs_dict = json.load(f)
    pairs = list(pairs_dict.items())
    df = pd.DataFrame(pairs, columns=['query_img', 'target_img'])
    df['query_path'] = 'Dataset/train/imgs/' + df['query_img']
    df['target_path'] = 'Dataset/train/imgs/' + df['target_img']

    all_img_paths = pd.unique(df[['query_path', 'target_path']].values.ravel())
    img_df = pd.DataFrame({'Path': all_img_paths})

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

    print("Creating embeddings for all unique images in training pairs...")
    img_df = make_artifacts(img_df, bioclip_args, dinov2_args, samvit_args)

    # Map embeddings back to pairs DataFrame
    emb_dict = dict(zip(img_df['Path'], img_df['embedding']))
    emb_dim = img_df['emb_dims'].iloc[0]
    df['query_embedding'] = df['query_path'].map(emb_dict)
    df['target_embedding'] = df['target_path'].map(emb_dict)
    df['emb_dims'] = emb_dim

    # Split into train/val pairs
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare test and submission sets as before (single images)
    print("Loading test data...")
    test_df = pd.read_json('Dataset/public/data.json')
    test_df["Path"] = test_df["Path"].str.replace("_image", "_Image", regex=False)
    test_df["Path"] = "Dataset/public/images/" + test_df["Path"].astype(str)
    print("Creating embeddings for test data...")
    test_df = make_artifacts(test_df, bioclip_args, dinov2_args, samvit_args)

    config = {
        'bioclip_args': bioclip_args,
        'dinov2_args': dinov2_args,
        'samvit_args': samvit_args,
        'embedding_dim': emb_dim
    }

    train_dataset = ENTRepDataset(train_df, pair_mode=True)
    val_dataset = ENTRepDataset(val_df, pair_mode=True)
    test_dataset = ENTRepDataset(test_df)

    submission_df = pd.read_csv('Dataset/test/i2i.csv', header=None, names=['Path'])
    submission_df['Path'] = 'Dataset/test/imgs/' + submission_df['Path'].astype(str)
    submission_df = make_artifacts(submission_df, bioclip_args, dinov2_args, samvit_args)
    submission_dataset = ENTRepDataset(submission_df)

    print("Saving artifacts...")
    save_artifacts(
        exp_name,
        train_dataset,
        val_dataset,
        test_dataset,
        config,
        submission_dataset
    )

    print("Validating artifacts by loading...")
    loaded_train_df, loaded_val_df, loaded_test_df = load_artifacts(exp_name)

    print(f"Original train shape: {train_df.shape}")
    print(f"Loaded train shape: {loaded_train_df.shape}")
    print(f"Original val shape: {val_df.shape}")
    print(f"Loaded val shape: {loaded_val_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    print(f"Loaded test shape: {loaded_test_df.shape}")

    print("Artifact creation and validation completed successfully!")
    test_entrep_dataset()
    return loaded_train_df, loaded_val_df, loaded_test_df

