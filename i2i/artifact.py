from pathlib import Path
import json
import numpy as np
import pandas as pd
from FAISS.combined_fe import CombinedFeatureExtractor
from tqdm import tqdm
import numbers
import ast

def list_converter(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, numbers.Integral):
        return int(obj)
    elif isinstance(obj, numbers.Real):
        return float(obj)
    else:
        return obj

def save_artifacts(
        exp_name: str, 
        train_dataset, 
        val_dataset, 
        test_dataset, 
        config,
        submission_dataset = None
    ):
    train_dataset.df.to_csv(f"results/train_df_{exp_name}.csv", index=None)
    val_dataset.df.to_csv(f"results/val_df_{exp_name}.csv", index=None)
    test_dataset.df.to_csv(f"results/test_df_{exp_name}.csv", index=None)
    
    # Handle train/val datasets with query_embedding/target_embedding
    if 'query_embedding' in train_dataset.df.columns and 'target_embedding' in train_dataset.df.columns:
        train_query_embeddings = np.array(train_dataset.df['query_embedding'].tolist())
        train_target_embeddings = np.array(train_dataset.df['target_embedding'].tolist())
        np.save(f"results/train_query_embedding_{exp_name}.npy", train_query_embeddings)
        np.save(f"results/train_target_embedding_{exp_name}.npy", train_target_embeddings)
    else:
        train_embeddings = np.array(train_dataset.df.embedding.tolist())
        np.save(f"results/train_numpy_embedding_{exp_name}.npy", train_embeddings)

    if 'query_embedding' in val_dataset.df.columns and 'target_embedding' in val_dataset.df.columns:
        val_query_embeddings = np.array(val_dataset.df['query_embedding'].tolist())
        val_target_embeddings = np.array(val_dataset.df['target_embedding'].tolist())
        np.save(f"results/val_query_embedding_{exp_name}.npy", val_query_embeddings)
        np.save(f"results/val_target_embedding_{exp_name}.npy", val_target_embeddings)
    else:
        val_embeddings = np.array(val_dataset.df.embedding.tolist())
        np.save(f"results/val_numpy_embedding_{exp_name}.npy", val_embeddings)

    # Test dataset (single embedding column)
    test_embeddings = np.array(test_dataset.df.embedding.tolist())
    np.save(f"results/test_numpy_embedding_{exp_name}.npy", test_embeddings)

    # Save submission_dataset if provided
    if submission_dataset is not None:
        submission_dataset.df.to_csv(f"results/submission_df_{exp_name}.csv", index=None)
        submission_embeddings = np.array(submission_dataset.df.embedding.tolist())
        np.save(f"results/submission_numpy_embedding_{exp_name}.npy", submission_embeddings)

    with open(f"results/config_{exp_name}.json", "w") as f:
        json.dump(convert_numpy(config), f, sort_keys=True, indent=4)

def load_artifacts(exp_name):


    # Try to use converters for possible columns
    train_df = pd.read_csv(
        f"results/train_df_{exp_name}.csv",
        converters={
            'query_embedding': list_converter,
            'target_embedding': list_converter,
            'embedding': list_converter,
            'query_path': str,
            'target_path': str
        }
    )
    val_df = pd.read_csv(
        f"results/val_df_{exp_name}.csv",
        converters={
            'query_embedding': list_converter,
            'target_embedding': list_converter,
            'embedding': list_converter,
            'query_path': str,
            'target_path': str
        }
    )
    test_df = pd.read_csv(
        f"results/test_df_{exp_name}.csv",
        converters={
            'embedding': list_converter,
            'Path': str
        }
    )

    # Try to load query/target embeddings for train/val, fallback to embedding
    try:
        train_query_embeddings = np.load(f"results/train_query_embedding_{exp_name}.npy", allow_pickle=True)
        train_target_embeddings = np.load(f"results/train_target_embedding_{exp_name}.npy", allow_pickle=True)
        train_df["query_embedding"] = [emb.tolist() for emb in train_query_embeddings]
        train_df["target_embedding"] = [emb.tolist() for emb in train_target_embeddings]
    except FileNotFoundError:
        train_embeddings = np.load(f"results/train_numpy_embedding_{exp_name}.npy", allow_pickle=True)
        train_df["embedding"] = [emb.tolist() for emb in train_embeddings]

    try:
        val_query_embeddings = np.load(f"results/val_query_embedding_{exp_name}.npy", allow_pickle=True)
        val_target_embeddings = np.load(f"results/val_target_embedding_{exp_name}.npy", allow_pickle=True)
        val_df["query_embedding"] = [emb.tolist() for emb in val_query_embeddings]
        val_df["target_embedding"] = [emb.tolist() for emb in val_target_embeddings]
    except FileNotFoundError:
        val_embeddings = np.load(f"results/val_numpy_embedding_{exp_name}.npy", allow_pickle=True)
        val_df["embedding"] = [emb.tolist() for emb in val_embeddings]

    test_embeddings = np.load(f"results/test_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    test_df["embedding"] = [emb.tolist() for emb in test_embeddings]
    
    return train_df, val_df, test_df

def make_artifacts(
        df: pd.DataFrame, 
        bioclip_args: dict, 
        dinov2_args: dict, 
        samvit_args: dict
    ):
    '''
    The input df has two columns: 'Path'(image path) and 'Classification' (image label)
    '''

    combined_extractor = CombinedFeatureExtractor(bioclip_args, dinov2_args, samvit_args)
    
    # Extract features for each image
    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = row['Path']
        
        # Extract combined features
        feature = combined_extractor.extract_feature(img_path).squeeze(0)
        embeddings.append(feature)
            
    embeddings_array = np.array(embeddings)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings_array.tolist()
    df['emb_dims'] = [embeddings_array.shape[1]] * len(df)
    
    print(f"Feature dimensions: {embeddings_array.shape[1]}")

    return df
