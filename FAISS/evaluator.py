import numpy as np
from collections import Counter

class Evaluator:
  @staticmethod
  def recall_at_k(query_labels: np.array, gallery_labels: np.array, topk_indices: np.array, k):
    """
    Calculate Recall@K for a retrieval system.
    
    Parameters:
        query_labels (np.ndarray): Labels of query images, shape (n_queries,)
        gallery_labels (np.ndarray): Labels of gallery images, shape (n_gallery,)
        topk_indices (np.ndarray): Indices of top-K gallery results for each query, shape (n_queries, K)
        k (int): Value of K for Recall@K
    
    Returns:
        float: Recall@K score
    """
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    topk_indices = np.asarray(topk_indices)[:, :k]

    correct_retrievals = 0
    
    # Pre-compute number of relevant images per query label in the gallery
    label_counts = Counter(gallery_labels)

    for i, query_label in enumerate(query_labels):
      # Number of relevant items in the gallery for this query
      num_relevant_in_gallery = label_counts[query_label]

      # Top-k retrieved labels for this query
      retrieved_labels = gallery_labels[topk_indices[i]]
      print(f"i = {i}, query_label = {query_label}: retrieved_labels {retrieved_labels}")
      num_relevant_in_topk = np.sum(retrieved_labels == query_label)

      # Recall for this query: relevant@k / relevant in dataset
      recall = num_relevant_in_topk / num_relevant_in_gallery if num_relevant_in_gallery > 0 else 0
      print(f"recall = {recall} = {num_relevant_in_topk} / {num_relevant_in_gallery}")
      correct_retrievals += recall

      return correct_retrievals / len(query_labels)

  @staticmethod
  def evaluate_recall_at_k(index, features, query_labels, K_values=[1, 5, 10]):
      recalls = {}

      # Perform FAISS search
      _, _, gallery_labels, _ = index.search(features, max(K_values))

      for k in K_values:
        recalls[k] = Evaluator.recall_at_k(
          query_labels=query_labels,
          gallery_labels=gallery_labels, 
          topk_indices=gallery_labels,
          k=k
        )

      return recalls
