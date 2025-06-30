from pathlib import Path
import json
import numpy as np
import pandas as pd
from FAISS.combined_fe import CombinedFeatureExtractor
from tqdm import tqdm
import numbers

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
        config
    ):
    train_dataset.df.to_csv(f"results/train_df_{exp_name}.csv", index=None)
    val_dataset.df.to_csv(f"results/val_df_{exp_name}.csv", index=None)
    test_dataset.df.to_csv(f"results/test_df_{exp_name}.csv", index=None)
    
    # Convert embedding lists to numpy arrays before saving
    train_embeddings = np.array(train_dataset.df.embedding.tolist())
    val_embeddings = np.array(val_dataset.df.embedding.tolist())
    test_embeddings = np.array(test_dataset.df.embedding.tolist())
    
    np.save(f"results/train_numpy_embedding_{exp_name}.npy", train_embeddings)
    np.save(f"results/val_numpy_embedding_{exp_name}.npy", val_embeddings)
    np.save(f"results/test_numpy_embedding_{exp_name}.npy", test_embeddings)


    with open(f"results/config_{exp_name}.json", "w") as f:
        json.dump(convert_numpy(config), f, sort_keys=True, indent=4)

def load_artifacts(exp_name):
    train_df = pd.read_csv(f"results/train_df_{exp_name}.csv")
    val_df = pd.read_csv(f"results/val_df_{exp_name}.csv")
    test_df = pd.read_csv(f"results/test_df_{exp_name}.csv")

    train_embeddings = np.load(f"results/train_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    val_embeddings = np.load(f"results/val_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    test_embeddings = np.load(f"results/test_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    
    # Convert numpy arrays back to lists for proper dataframe storage
    train_df["embedding"] = [emb.tolist() for emb in train_embeddings]
    val_df["embedding"] = [emb.tolist() for emb in val_embeddings]
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
