import faiss
import numpy as np

class FAISSIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.labels = []
        self.paths = []

    def add_features(self, features, labels, paths):
        self.index.add(features)
        self.labels.extend(labels)
        self.paths.extend(paths)

    def search(self, features, k):
        distances, indices = self.index.search(features, k)
        retrieved_labels = np.array([self.labels[i] for i in indices.flatten()]).reshape(indices.shape)
        retrieved_paths = np.array([self.paths[i] for i in indices.flatten()]).reshape(indices.shape)
        return distances, indices, retrieved_labels, retrieved_paths
