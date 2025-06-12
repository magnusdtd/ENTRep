import torch
from PIL import Image
import open_clip
import time

def main():
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

if __name__ == "__main__":
  start_time = time.time()
  main()
  end_time = time.time()
  print(f"Time consumption: {end_time - start_time:.2f} seconds")
