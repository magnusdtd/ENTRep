import os
import json
import zipfile
import pandas as pd
import datetime
from DenseNet.inference import preprocess_image, classify_image

def make_submission(model, device:str, test_file_path: str, output_folder_path: str = './results'):
    # Load test file paths
    test_df = pd.read_csv(test_file_path, header=None, names=['image_path'])

    # Create predictions dictionary
    predictions = {}
    for img_name in test_df['image_path']:
        image_path = os.path.join('Dataset/test/imgs', img_name)
        image_tensor = preprocess_image(image_path)
        predicted_label = classify_image(model, image_tensor, device)
        predictions[image_path] = predicted_label

    # Generate unique JSON filename
    daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file_name = f'result_{daytime}.json'
    json_file_path = os.path.join(output_folder_path, json_file_name)

    # Save predictions to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(predictions, json_file)

    # Create ZIP archive
    zip_file_path = os.path.join(output_folder_path, 'predictions.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        zip_file.write(json_file_path, arcname=json_file_name)

    print(f"Submission file created at: {zip_file_path}")