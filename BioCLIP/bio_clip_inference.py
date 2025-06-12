import torch
from PIL import Image
import open_clip
import time
import os

def cls_inference():
  device = torch.device("cuda:x" if torch.cuda.is_available() else "cpu")
  model_path = "pure_bioclip/open_clip_pytorch_model.bin"
  model_name = "hf-hub:imageomics/bioclip"
  model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
  tokenizer = open_clip.get_tokenizer(model_name)

  model.eval()

  labels = [
    "nose-right", 
    "nose-left" , 
    "ear-right" , 
    "ear-left"  , 
    "vc-open"   , 
    "vc-closed" , 
    "throat"    , 
  ]
  img_path = "Dataset/train/imgs/078c91ff-9899-436c-854c-4227df8c1229.png"
  image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(device)
  text = tokenizer(labels).to(device)

  with torch.no_grad():
      if torch.cuda.is_available():
          with torch.autocast("cuda"):
              image_features = model.encode_image(image)
              text_features = model.encode_text(text)
      else:
          image_features = model.encode_image(image)
          text_features = model.encode_text(text)

      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)

      text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

  for label, prob in zip(labels, text_probs.squeeze().tolist()):
      print(f"Label: {label}, Probability: {prob}")

  # Get the index of the best matching label(s)
  best_match_idx = text_probs.argmax(dim=-1)
  best_labels = [labels[idx] for idx in best_match_idx]
  print("Best match labels:", best_labels)

  Image.open(img_path).show()

def t2i_inference():
  device = torch.device("cuda:x" if torch.cuda.is_available() else "cpu")
  model_path = "pure_bioclip/open_clip_pytorch_model.bin"
  model_name = "hf-hub:imageomics/bioclip"

  text_queries = [
    "edema and erythema of the arytenoid cartilages"
  ]
  image_folder_path = "Dataset/train/imgs"

  # Load BioCLIP model and tokenizer
  model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
  tokenizer = open_clip.get_tokenizer(model_name)

  # Preprocess text queries
  text_tokens = tokenizer(text_queries).to(device)

  # Extract image features
  image_features_dict = {}
  for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    image_tensor = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
      if torch.cuda.is_available():
        with torch.autocast("cuda"):
          image_features = model.encode_image(image_tensor)
      else:
          image_features = model.encode_image(image_tensor)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features_dict[image_name] = image_features

  # Match text queries to images
  predictions = {}
  with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for i, text_query in enumerate(text_queries):
      similarities = {}
      for image_name, image_features in image_features_dict.items():
        similarity = (100.0 * text_features[i] @ image_features.T).item()
        similarities[image_name] = similarity

      best_match_image = max(similarities, key=similarities.get)
      predictions[text_query] = best_match_image

  print(predictions)

if __name__ == "__main__":
  start_time = time.time()
  # cls_inference()
  t2i_inference()
  end_time = time.time()
  print(f"Time consumption: {end_time - start_time:.2f} seconds")
