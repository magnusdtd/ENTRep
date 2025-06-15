import os
import json
import zipfile
import pandas as pd
import datetime
from torch.utils.data import DataLoader
from FAISS.dataset import ENTRepDataset
from FAISS.transform import get_transform
from FAISS.faiss_indexer import FAISSIndexer
from FAISS.feature_extractor import FeatureExtractor

def make_submission(feature_extractor: FeatureExtractor, model_name: str, test_file_path: str, output_folder_path: str = './results'):
    # Load test dataset
    test_df = pd.read_csv(test_file_path, header=None, names=['Path'])
    dataset = ENTRepDataset(
        test_df, {}, 
        get_transform(train=False),
        images_dir='Dataset/test/imgs',
        is_inference=True
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    # Extract features
    features, paths = feature_extractor.extract_features(dataloader, True)

    # Build FAISS index
    dim = features.shape[1]
    indexer = FAISSIndexer(dim)
    indexer.add_features(features, [None] * len(features), paths)

    # Retrieve nearest neighbors
    retrieved_images = {}
    for i, path in enumerate(paths):
        _, _, _, retrieved_paths = indexer.search(features[i:i+1], k=2)
        retrieved_images[path] = retrieved_paths[0][1]

    # Save to JSON
    daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file_name = f'{model_name}_{daytime}.json'
    json_file_path = os.path.join(output_folder_path, json_file_name)
    with open(json_file_path, 'w') as json_file:
        json.dump(retrieved_images, json_file, indent=4)

    # Compress JSON into ZIP
    zip_file_path = os.path.join(output_folder_path, f'{model_name}_{daytime}.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        zip_file.write(json_file_path, arcname=json_file_name)

    print(f"Submission saved to {zip_file_path}")