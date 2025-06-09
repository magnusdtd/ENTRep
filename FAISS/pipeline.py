import pandas as pd
from FAISS.dataset import ENTRepDataset
from torch.utils.data import DataLoader
from FAISS.transform import get_transform
from FAISS.faiss_indexer import FAISSIndexer
from FAISS.ResNet import ResNet_FE
from FAISS.evaluator import Evaluator

class Pipeline:
  def __init__(self, dataset_path:str, class_feature_map:dict, backbone, model_path:str, batch_size=4, num_workers=4):
    self.dataset_path = dataset_path
    self.class_feature_map = class_feature_map
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.backbone = backbone
    self.model_path = model_path

  def run(self):
    # Load dataset
    df = pd.read_json(self.dataset_path, orient='index').reset_index()
    df.columns = ['Path', 'Classification']
    dataset = ENTRepDataset(df, self.class_feature_map, get_transform(train=False))
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Extract features
    feature_extractor = ResNet_FE(self.backbone)
    feature_extractor.load_model_state(self.model_path, self.backbone)
    features, labels, paths = feature_extractor.extract_features(dataloader)

    # Build FAISS index
    dim = features.shape[1]
    indexer = FAISSIndexer(dim)
    indexer.add_features(features)

    # Evaluate recall
    evaluator = Evaluator()
    K_values = [1, 5, 10]
    avg_recalls = evaluator.evaluate_recall_at_k(indexer.index, features, labels, paths, K_values)

    # Print results
    for K, val in avg_recalls.items():
        print(f"Recall@{K}: {val:.4f}")
