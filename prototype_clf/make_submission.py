import os
import json
import zipfile
import datetime
import torch
import pandas as pd
import numpy as np

def make_submission(
        prototype_classifier, 
        exp_name: str, 
        output_folder_path: str = './results'
    ):
    # Load test file
    test_df = pd.read_csv(f"results/test_df_{exp_name}.csv")
    test_embeddings = np.load(f"results/test_numpy_embedding_{exp_name}.npy", allow_pickle=True)
    test_df["embedding"] = [emb.tolist() for emb in test_embeddings]

    # Create predictions dictionary
    predictions = {}
    for _, row in test_df.iterrows():
        img_path = row['Path']
        embedding = row['embedding']
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(prototype_classifier.device)
        with torch.no_grad():
            probas = prototype_classifier.make_prediction(embedding_tensor)
            pred_class = torch.argmax(probas, dim=1).item()
        img_name = os.path.basename(img_path)
        predictions[img_name] = pred_class

    # Generate unique JSON filename with exp_name as prefix
    daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file_name = f'{exp_name}_{daytime}.json'
    json_file_path = os.path.join(output_folder_path, json_file_name)

    # Save predictions to JSON file
    os.makedirs(output_folder_path, exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(predictions, json_file)

    # Create ZIP archive with the same name as the JSON file
    zip_file_path = os.path.join(output_folder_path, f'{exp_name}_{daytime}.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        zip_file.write(json_file_path, arcname=json_file_name)

    print(f"Submission file created at: {zip_file_path}")