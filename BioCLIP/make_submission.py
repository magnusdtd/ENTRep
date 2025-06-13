import torch
from PIL import Image
import open_clip
import os
import json
import zipfile
import pandas as pd
import datetime

def make_submission_cls_task(
    model_name: str, 
    model_path:str, 
    test_file_path: str, 
    labels_map:dict[str, int], 
    output_folder_path: str = './results'
  ):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  test_df = pd.read_csv(test_file_path, header=None, names=['Path'])

  # Load BioCLIP model and tokenizer
  model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
  model.to(device)
  model.eval()
  tokenizer = open_clip.get_tokenizer(model_name)

  label_names = list(labels_map.keys())
  predictions = {}
  for img_name in test_df['Path']:
    image_path = os.path.join('Dataset/test/imgs', img_name)
    image_tensor = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(label_names).to(device)

    with torch.no_grad():
      image_features = model.encode_image(image_tensor)
      text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    best_match_idx = text_probs.argmax(dim=-1)
    predictions[img_name] = labels_map[label_names[best_match_idx.item()]]

  # Generate unique JSON filename with model_name as prefix
  daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  json_file_name = f'BioCLIP_cls_{daytime}.json'
  json_file_path = os.path.join(output_folder_path, json_file_name)

  # Save predictions to JSON file
  with open(json_file_path, 'w') as json_file:
    json.dump(predictions, json_file)

  # Create ZIP archive with the same name as the JSON file
  zip_file_path = os.path.join(output_folder_path, f'BioCLIP_cls_{daytime}.zip')
  with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
    zip_file.write(json_file_path, arcname=json_file_name)

  print(f"Submission file created at: {zip_file_path}")

def make_submission_t2i_task(
    model_name: str, 
    model_path: str, 
    test_file_path: list, 
    image_folder_path: str, 
    output_folder_path: str = './results'
  ):
  test_df = pd.read_csv(test_file_path, header=None, names=['Query'])

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
  model.to(device)
  model.eval()
  tokenizer = open_clip.get_tokenizer(model_name)

  # Preprocess text queries
  text_tokens = tokenizer(test_df['Query'].to_list()).to(device)

  # Extract image features
  image_features_dict = {}
  for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    image_tensor = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
      image_features = model.encode_image(image_tensor)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features_dict[image_name] = image_features

  # Match text queries to images
  predictions = {}
  with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for i, text_query in enumerate(test_df['Query']):
      similarities = {}
      for image_name, image_features in image_features_dict.items():
        similarity = (100.0 * text_features[i] @ image_features.T).item()
        similarities[image_name] = similarity

      best_match_image = max(similarities, key=similarities.get)
      predictions[text_query] = best_match_image

  # Generate unique JSON filename with model_name as prefix
  daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  json_file_name = f'BioCLIP_t2i_{daytime}.json'
  json_file_path = os.path.join(output_folder_path, json_file_name)

  # Save predictions to JSON file
  with open(json_file_path, 'w') as json_file:
    json.dump(predictions, json_file)

  # Create ZIP archive with the same name as the JSON file
  zip_file_path = os.path.join(output_folder_path, f'BioCLIP_t2i_{daytime}.zip')
  with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
    zip_file.write(json_file_path, arcname=json_file_name)

  print(f"Submission file created at: {zip_file_path}")
