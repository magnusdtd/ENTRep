from pathlib import Path
import json
import numpy as np
import pandas as pd
from FAISS.combined_fe import CombinedFeatureExtractor
from tqdm import tqdm

def save_artifacts(
        exp_name: str, 
        train_dataset, 
        val_dataset, 
        test_dataset, 
        config
    ):
    embed_dims = test_dataset.df.emb_dims.iloc[0]
    np.save(f"results/numpy_embed_dims_{exp_name}.npy", embed_dims)
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
        json.dump(config, f, sort_keys=True, indent=4)

def load_artifacts(exp_name):
    train_df = pd.read_csv(f"results/train_df_{exp_name}.csv")
    val_df = pd.read_csv(f"results/val_df_{exp_name}.csv")
    test_df = pd.read_csv(f"results/test_df_{exp_name}.csv")
    embed_dims = np.load(f"results/numpy_embed_dims_{exp_name}.npy", allow_pickle=True)
    train_df['embed_dims'] = train_df.apply(lambda row: embed_dims, axis=1)
    val_df['embed_dims'] = val_df.apply(lambda row: embed_dims, axis=1)
    test_df['embed_dims'] = test_df.apply(lambda row: embed_dims, axis=1)
    train_embeddings = np.load(f"results/train_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    val_embeddings = np.load(f"results/val_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    test_embeddings = np.load(f"results/test_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    train_df["embedding"] = train_embeddings
    val_df["embedding"] = val_embeddings
    test_df["embedding"] = test_embeddings
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
        feature = combined_extractor.extract_feature(img_path)
        embeddings.append(feature)
            
    embeddings_array = np.array(embeddings)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings_array.tolist()
    df['emb_dims'] = [embeddings_array.shape[1]] * len(df)
    
    print(f"Feature dimensions: {embeddings_array.shape[1]}")

    return df