import faiss

class FAISSIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add_features(self, features):
        self.index.add(features)

    def search(self, features, k):
        return self.index.search(features, k)
