from PIL import Image
import pandas as pd
import torch
import open_clip
import os

class TextToImageEvaluator:
  def __init__(
      self, 
      df: pd.DataFrame, 
      queries: dict[str], 
      model_name: str, 
      model_path: str,
      path_column: str,
      caption_column: str
    ):
    '''
    The df contains paths to images at column 'Path'.
    '''
    self.df = df
    self.model_name = model_name
    self.model_path = model_path
    self.queries = queries
    self.path_column = path_column
    self.caption_column = caption_column
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    if self.model_path:
      self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name, pretrained=self.model_path)
    else:
      self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name)
    self.model.to(self.device)
    self.model.eval()
    self.tokenizer = open_clip.get_tokenizer(self.model_name)
    self.query_tokens = self.tokenizer(queries).to(self.device)

    self.image_features_dict = {}
    self._feature_extract()

  def _feature_extract(self):
    for _, row in self.df.iterrows():
      image_path = row[self.path_column]
      image_name = os.path.basename(image_path)
      image_tensor = self.preprocess_val(Image.open(image_path)).unsqueeze(0).to(self.device)

      with torch.no_grad():
        image_features = self.model.encode_image(image_tensor)

      image_features /= image_features.norm(dim=-1, keepdim=True)
      self.image_features_dict[image_name] = image_features

  def get_recall_at_k(self, k: int):
    num_queries_with_correct_image_in_top_k = 0
    total_num_queries = len(self.queries)

    # Debug
    top_k_dict = {}

    with torch.no_grad():
      text_features = self.model.encode_text(self.query_tokens)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      text_features = text_features.float()
      for i, query in enumerate(self.queries):
        similarities = {}
        for image_name, image_features in self.image_features_dict.items():
          similarity = (100.0 * text_features[i] @ image_features.T).item()
          similarities[image_name] = similarity

        # Sort images by similarity and get top-k
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_images = [image[0] for image in sorted_images[:k]]

        # Check if the correct image is in the top-k images
        if query in top_k_images:
          num_queries_with_correct_image_in_top_k += 1

        top_k_dict[query] = top_k_images

    for query, top_k_images in top_k_dict.items():
      print(f"{query}, {top_k_images}")

    recall_at_k = num_queries_with_correct_image_in_top_k / total_num_queries
    return recall_at_k
