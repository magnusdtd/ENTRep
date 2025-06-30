import numpy as np
from i2i.artifact import load_artifacts
from i2i.ENTRepDataset import ENTRepDataset

def test_entrep_dataset():
    print("Loading artifacts...")
    try:
        train_df, val_df, test_df = load_artifacts("proto_clf_DINOv2s_BioCLIP_SAMViTB")
        print("✓ Artifacts loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load artifacts: {e}")
        return
    
    print(f"\nDataframe shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Val: {val_df.shape}")
    print(f"Test: {test_df.shape}")
    
    print(f"\nColumns in train_df: {list(train_df.columns)}")
    
    # Check if embeddings are properly loaded
    print(f"\nChecking embeddings...")
    if 'query_embedding' in train_df.columns and 'target_embedding' in train_df.columns:
        train_query_emb_sample = train_df['query_embedding'].iloc[0]
        train_target_emb_sample = train_df['target_embedding'].iloc[0]
        print(f"✓ query_embedding and target_embedding found in train_df")
        print(f"  Sample query_embedding type: {type(train_query_emb_sample)} length: {len(train_query_emb_sample)}")
        print(f"  Sample target_embedding type: {type(train_target_emb_sample)} length: {len(train_target_emb_sample)}")
    elif 'embedding' in train_df.columns:
        train_emb_sample = train_df['embedding'].iloc[0]
        print(f"✓ Embeddings found in train_df")
        print(f"  Sample embedding type: {type(train_emb_sample)} length: {len(train_emb_sample)}")
    else:
        print("✗ No embeddings found in train_df")
        return
    
    # Create ENTRep datasets
    print(f"\nCreating ENTRep datasets...")
    try:
        train_dataset = ENTRepDataset(train_df, split='train', pair_mode=True)
        val_dataset = ENTRepDataset(val_df, split='val', pair_mode=True)
        test_dataset = ENTRepDataset(test_df, split='test', pair_mode=False)
        print("✓ ENTRep datasets created successfully")
    except Exception as e:
        print(f"✗ Failed to create ENTRep datasets: {e}")
        return
    
    print(f"\nDataset lengths:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Test __getitem__ method for each dataset
    print(f"\nTesting __getitem__ method...")
    
    # Test train dataset (pair mode)
    print(f"\n--- Train Dataset Test ---")
    try:
        query_img, target_img, query_emb, target_emb, query_path, target_path = train_dataset[0]
        print(f"✓ Train __getitem__ successful")
        print(f"  Query image type: {type(query_img)}")
        print(f"  Target image type: {type(target_img)}")
        print(f"  Query embedding type: {type(query_emb)}, shape: {query_emb.shape}")
        print(f"  Target embedding type: {type(target_emb)}, shape: {target_emb.shape}")
        print(f"  Query path: {query_path}")
        print(f"  Target path: {target_path}")
    except Exception as e:
        print(f"✗ Train __getitem__ failed: {e}")
    
    # Test validation dataset (pair mode)
    print(f"\n--- Validation Dataset Test ---")
    try:
        query_img, target_img, query_emb, target_emb, query_path, target_path = val_dataset[0]
        print(f"✓ Val __getitem__ successful")
        print(f"  Query image type: {type(query_img)}")
        print(f"  Target image type: {type(target_img)}")
        print(f"  Query embedding type: {type(query_emb)}, shape: {query_emb.shape}")
        print(f"  Target embedding type: {type(target_emb)}, shape: {target_emb.shape}")
        print(f"  Query path: {query_path}")
        print(f"  Target path: {target_path}")
    except Exception as e:
        print(f"✗ Val __getitem__ failed: {e}")
    
    # Test test dataset (single mode)
    print(f"\n--- Test Dataset Test ---")
    try:
        image, file_path, emb = test_dataset[0]
        print(f"✓ Test __getitem__ successful")
        print(f"  Image type: {type(image)}")
        print(f"  File path: {file_path}")
        print(f"  Embedding type: {type(emb)}")
        if emb is not None:
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding dtype: {emb.dtype}")
        else:
            print("  ✗ Embedding is None")
    except Exception as e:
        print(f"✗ Test __getitem__ failed: {e}")
    
    print(f"\n=== Test completed ===")

if __name__ == "__main__":
    test_entrep_dataset() 