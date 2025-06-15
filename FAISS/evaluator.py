import numpy as np

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

    for i, query_label in enumerate(query_labels):
      # Top-k retrieved labels for this query
      retrieved_labels = gallery_labels[topk_indices[i]]
      num_relevant_in_topk = np.sum(retrieved_labels == query_label)

      # Recall for this query: relevant@k / total images in query
      recall = num_relevant_in_topk / k
      correct_retrievals += recall

    return correct_retrievals / len(query_labels)

  @staticmethod
  def evaluate_recall_at_k(index, features, query_labels, K_values=[1, 5, 10]):
      recalls = {}

      # Perform FAISS search
      _, indices, _, _ = index.search(features, max(K_values))

      for k in K_values:
        recalls[k] = Evaluator.recall_at_k(
          query_labels=query_labels,    # (n_queries,) = (1291,)
          gallery_labels=index.labels,  # (n_gallery,) = (1291,)
          topk_indices=indices,         # (n_queries, K) = (1291, K)
          k=k
        )

      return recalls