import faiss
import numpy as np
from typing import List, Tuple

class FAISSIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.labels = []
        self.paths = []

    def add_features(
        self,
        features: np.ndarray,
        labels: List[str],
        paths: List[str]
    ) -> None:
        self.index.add(features)
        self.labels.extend(labels)
        self.paths.extend(paths)

    def search(
        self,
        features: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances, indices = self.index.search(features, k)
        retrieved_labels = np.array([self.labels[i] for i in indices.flatten()]).reshape(indices.shape)
        retrieved_paths = np.array([self.paths[i] for i in indices.flatten()]).reshape(indices.shape)
        return distances, indices, retrieved_labels, retrieved_paths
