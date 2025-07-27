import pandas as pd
from FAISS.dataset import ENTRepDataset
from torch.utils.data import DataLoader
from FAISS.transform import get_transform
from FAISS.faiss_indexer import FAISSIndexer
from FAISS.evaluator import Evaluator
from FAISS.feature_extractor import FeatureExtractor

class Pipeline:
  def __init__(
      self, 
      train_dataset_path: str,
      test_dataset_path: str,
      class_feature_map: dict, 
      feature_extractor: FeatureExtractor, 
      batch_size=4, 
      num_workers=4
    ):
    self.train_dataset_path = train_dataset_path
    self.test_dataset_path = test_dataset_path
    self.class_feature_map = class_feature_map
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.feature_extractor = feature_extractor

  def run(self):
    # Load train dataset
    train_df = pd.read_json(self.train_dataset_path, orient='index').reset_index()
    train_df.columns = ['Path', 'Classification']
    train_df["Path"] = "Dataset/train/imgs/" + train_df["Path"].astype(str)
    train_dataset = ENTRepDataset(train_df, self.class_feature_map, get_transform(train=False))
    train_dataloader = DataLoader(
      train_dataset, 
      batch_size=self.batch_size, 
      shuffle=False, 
      num_workers=self.num_workers
    )

    # Extract train features
    train_features, labels, paths = self.feature_extractor.extract_features(train_dataloader)

    # Build FAISS index
    dim = train_features.shape[1]
    indexer = FAISSIndexer(dim)
    indexer.add_features(train_features, labels, paths)

    # Load test dataset
    test_df = pd.read_json(self.test_dataset_path)
    test_df["Path"] = test_df["Path"].str.replace("_image", "_Image", regex=False)
    test_df["Path"] = "Dataset/public/images/" + test_df["Path"].astype(str)
    test_dataset = ENTRepDataset(
        test_df, 
        self.class_feature_map,
        transform=get_transform(train=False)
    )
    test_dataloader = DataLoader(
      test_dataset, 
      batch_size=self.batch_size, 
      shuffle=False, 
      num_workers=self.num_workers
    )

    # Extract test features
    test_features, test_labels, _ = self.feature_extractor.extract_features(test_dataloader)

    # Evaluate recall
    evaluator = Evaluator()
    K_values = [1, 5, 10]
    avg_recalls = evaluator.evaluate_recall_at_k(indexer, test_features, test_labels, K_values)

    for K, val in avg_recalls.items():
      print(f"Recall@{K}: {val}")
