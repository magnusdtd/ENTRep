from dinov2.compute_embeddings import DinoV2
from dinov2.svm import SVM
import torch
import torchvision.transforms as T
import pandas as pd

def main():
  dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

  transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

  dinov2 = DinoV2(dinov2_vits14, "Dataset/train/imgs", transform_image)
  embeddings = dinov2.compute_embeddings("Dataset/train/cls.json")
  embedding_list = list(embeddings.values())

  df = pd.read_json('Dataset/train/cls.json', orient='index')
  df = df.reset_index()
  df.columns = ['Path', 'Ground Truth Label']
  y = df['Ground Truth Label'].to_list()
  labels = [
      "nose-right", 
      "nose-left" , 
      "ear-right" , 
      "ear-left"  , 
      "vc-open"   , 
      "vc-closed" , 
      "throat"    , 
  ]

  label_map = {
      "nose-right": 0, 
      "nose-left" : 1, 
      "ear-right" : 2, 
      "ear-left"  : 3, 
      "vc-open"   : 4, 
      "vc-closed" : 5, 
      "throat"    : 6, 
  }

  dinov2_svm = SVM(dinov2_vits14, embedding_list, y, transform_image)
  dinov2_svm.predict("Dataset/test/imgs/0a653eb8-fa96-47d4-9e32-eeb1792140d3.png")

  dinov2_svm.make_submission('Dataset/test/cls.csv', 'Dataset/test/imgs', label_map)

if __name__ == "__main__":
  main()

