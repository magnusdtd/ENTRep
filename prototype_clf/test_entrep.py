import numpy as np
from prototype_clf.artifact import load_artifacts
from prototype_clf.ENTRepDataset import ENTRepDataset

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
    if 'embedding' in train_df.columns:
        train_emb_sample = train_df['embedding'].iloc[0]
        print(f"✓ Embeddings found in train_df")
        print(f"  Sample embedding type: {type(train_emb_sample)}")
        if isinstance(train_emb_sample, list):
            print(f"  Sample embedding length: {len(train_emb_sample)}")
        elif isinstance(train_emb_sample, np.ndarray):
            print(f"  Sample embedding shape: {train_emb_sample.shape}")
    else:
        print("✗ No embeddings found in train_df")
        return
    
    # Create ENTRep datasets
    print(f"\nCreating ENTRep datasets...")
    try:
        train_dataset = ENTRepDataset(train_df, split='train')
        val_dataset = ENTRepDataset(val_df, split='val')
        test_dataset = ENTRepDataset(test_df, split='test')
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
    
    # Test train dataset
    print(f"\n--- Train Dataset Test ---")
    try:
        image, label, file_path, emb = train_dataset[0]
        print(f"✓ Train __getitem__ successful")
        print(f"  Image type: {type(image)}")
        print(f"  Label: {label} (type: {type(label)})")
        print(f"  File path: {file_path}")
        print(f"  Embedding type: {type(emb)}")
        if emb is not None:
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding dtype: {emb.dtype}")
        else:
            print("  ✗ Embedding is None")
    except Exception as e:
        print(f"✗ Train __getitem__ failed: {e}")
    
    # Test validation dataset
    print(f"\n--- Validation Dataset Test ---")
    try:
        image, label, file_path, emb = val_dataset[0]
        print(f"✓ Val __getitem__ successful")
        print(f"  Image type: {type(image)}")
        print(f"  Label: {label} (type: {type(label)})")
        print(f"  File path: {file_path}")
        print(f"  Embedding type: {type(emb)}")
        if emb is not None:
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding dtype: {emb.dtype}")
        else:
            print("  ✗ Embedding is None")
    except Exception as e:
        print(f"✗ Val __getitem__ failed: {e}")
    
    # Test test dataset
    print(f"\n--- Test Dataset Test ---")
    try:
        image, label, file_path, emb = test_dataset[0]
        print(f"✓ Test __getitem__ successful")
        print(f"  Image type: {type(image)}")
        print(f"  Label: {label} (type: {type(label)})")
        print(f"  File path: {file_path}")
        print(f"  Embedding type: {type(emb)}")
        if emb is not None:
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding dtype: {emb.dtype}")
        else:
            print("  ✗ Embedding is None")
    except Exception as e:
        print(f"✗ Test __getitem__ failed: {e}")
    
    # Test get_embeddings_for_class method
    print(f"\n--- Testing get_embeddings_for_class ---")
    try:
        class_embeddings = train_dataset.get_embeddings_for_class(0)
        print(f"✓ get_embeddings_for_class successful")
        print(f"  Number of samples for class 0: {len(class_embeddings)}")
        if len(class_embeddings) > 0:
            sample_emb = class_embeddings.iloc[0]
            print(f"  Sample embedding type: {type(sample_emb)}")
            if isinstance(sample_emb, list):
                print(f"  Sample embedding length: {len(sample_emb)}")
            elif isinstance(sample_emb, np.ndarray):
                print(f"  Sample embedding shape: {sample_emb.shape}")
    except Exception as e:
        print(f"✗ get_embeddings_for_class failed: {e}")
    
    print(f"\n=== Test completed ===")

if __name__ == "__main__":
    test_entrep_dataset() 