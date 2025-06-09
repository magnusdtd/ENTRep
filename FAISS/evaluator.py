import numpy as np
import pandas as pd

class Evaluator:
  @staticmethod
  def evaluate_recall_at_k(index, features, labels, paths, df, K_values=[1, 5, 10]):
    recalls = {K: [] for K in K_values}
    total_per_class = {}

    # Calculate total relevant images per class
    for label in labels:
      total_per_class[label] = total_per_class.get(label, 0) + 1

    # Perform FAISS search
    _, I = index.search(features, max(K_values) + 1)

    for i, query_lbl in enumerate(labels):
      query_class = df.loc[df['Path'] == paths[i], 'Classification'].values[0]
      retrieved_idxs = I[i].tolist()

      # Exclude the query image itself from the retrieved results
      if retrieved_idxs[0] == i:
        retrieved_idxs = retrieved_idxs[1:]
      else:
        retrieved_idxs = retrieved_idxs[:-1]

      for K in K_values:
        topk = retrieved_idxs[:K]
        rel = sum(1 for idx in topk if df.loc[df['Path'] == paths[idx], 'Classification'].values[0] == query_class)
        denominator = total_per_class[query_lbl] - 1
        if denominator == 0:
            recall = 0
        else:
            recall = rel / denominator
        recalls[K].append(recall)

    avg_recalls = {K: np.mean(recalls[K]) for K in K_values}
    return avg_recalls
