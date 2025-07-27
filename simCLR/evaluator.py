import torch
from simCLR.transform import get_transform
import torch.nn.functional as F
from tqdm import tqdm
from simCLR.dataset import ENTRepDataset
from torch.utils.data import DataLoader
from PIL import Image

class SimCLREvaluator:
    def __init__(self, model, loader: DataLoader) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = loader
        self.model.to(self.device)
        self.model.eval()

        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for xi, xj in tqdm(self.loader, desc="Extracting features"):
                xi, xj = xi.to(self.device), xj.to(self.device)
                zi = self.model(xi)
                zj = self.model(xj)
                embeddings.append(zi.detach().cpu())
                embeddings.append(zj.detach().cpu())
        
        # Concatenate all embeddings
        self.embeddings = torch.cat(embeddings, dim=0)
        self.embeddings = F.normalize(self.embeddings, dim=1)
        self.N = self.embeddings.size(0)
        self.num_pairs = self.N // 2

    def get_recall_k(self, top_k=5):
        """Compute Recall@k for positive pairs"""
        similarity_matrix = torch.mm(self.embeddings, self.embeddings.t())
        recall_at_k = 0
        for pair_idx in range(self.num_pairs):
            i = pair_idx * 2
            j = i + 1
            # For i, positive is j
            for anchor, positive in [(i, j), (j, i)]:
                sim_scores = similarity_matrix[anchor].clone()
                sim_scores[anchor] = -float('inf')  # Exclude self
                sorted_indices = torch.argsort(sim_scores, descending=True)
                # Find where the positive pair is in the ranking
                rank = (sorted_indices == positive).nonzero(as_tuple=True)[0].item() + 1
                if rank <= top_k:
                    recall_at_k += 1
        
        recall_at_k = recall_at_k / self.N
        return recall_at_k
    
    def get_mrr(self):
        """Compute Mean Reciprocal Rank (MRR) for positive pairs"""
        similarity_matrix = torch.mm(self.embeddings, self.embeddings.t())
        reciprocal_ranks = []
        for pair_idx in range(self.num_pairs):
            i = pair_idx * 2
            j = i + 1
            # For i, positive is j
            for anchor, positive in [(i, j), (j, i)]:
                sim_scores = similarity_matrix[anchor].clone()
                sim_scores[anchor] = -float('inf')  # Exclude self
                sorted_indices = torch.argsort(sim_scores, descending=True)
                # Find where the positive pair is in the ranking
                rank = (sorted_indices == positive).nonzero(as_tuple=True)[0].item() + 1
                reciprocal_ranks.append(1.0 / rank)
        
        mrr = sum(reciprocal_ranks) / self.N
        return mrr