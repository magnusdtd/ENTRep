from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer
import torch

class ImageTextRetrievalEvaluator(SentenceEvaluator):
  def __init__(
    self,
    images: list,
    texts: list[str],
    name: str = '',
    k: int = 1,
    batch_size: int = 32,
    show_progress_bar: bool = True
  ):
    self.images = images
    self.texts = texts
    self.name = name
    self.k = k
    self.batch_size = batch_size
    self.show_progress_bar = show_progress_bar

  def __call__(self,
    model: SentenceTransformer,
    output_path: str = None,
    epoch: int = -1,
    steps: int = -1
  ) -> dict[str, float]:
    
    # Get embeddings for all images
    img_embeddings = model.encode(
        self.images,
        batch_size=self.batch_size,
        show_progress_bar=self.show_progress_bar,
        convert_to_tensor=True
    )
    
    # Get embeddings for all texts
    text_embeddings = model.encode(
        self.texts,
        batch_size=self.batch_size,
        show_progress_bar=self.show_progress_bar,
        convert_to_tensor=True
    )
    
    # Compute similarity matrix
    cos_scores = torch.nn.functional.cosine_similarity(
        img_embeddings.unsqueeze(1),
        text_embeddings.unsqueeze(0),
        dim=2
    )
    
    # Get indices of top k predictions for each image
    _, top_indices = torch.topk(cos_scores, k=self.k, dim=1)
    
    # Calculate Recall@k (correct if ground truth index is in top k predictions)
    correct = sum(i in top_indices[i].tolist() for i in range(len(self.images)))
    recall_at_k = correct / len(self.images)

    return {f'{self.name}_Recall@{self.k}': recall_at_k}