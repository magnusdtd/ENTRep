import os
import json
import zipfile
import pandas as pd
import datetime
from classification.inference import preprocess_image, classify_image

def make_submission(model, model_name:str, device:str, df: pd.DataFrame, output_folder_path: str = './results'):
    # Create predictions dictionary
    predictions = {}
    for img_path in df['Path']:
        img_tensor = preprocess_image(img_path)
        predicted_label = classify_image(model, img_tensor, device)
        img_name = os.path.basename(img_path)
        predictions[img_name] = predicted_label

    # Generate unique JSON filename with model_name as prefix
    daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file_name = f'{model_name}_{daytime}.json'
    json_file_path = os.path.join(output_folder_path, json_file_name)

    # Save predictions to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(predictions, json_file)

    # Create ZIP archive with the same name as the JSON file
    zip_file_path = os.path.join(output_folder_path, f'{model_name}_{daytime}.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        zip_file.write(json_file_path, arcname=json_file_name)

    print(f"Submission file created at: {zip_file_path}")