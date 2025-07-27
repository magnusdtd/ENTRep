import torch

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, zi, zj):
        """
        NT-Xent loss computes similarity between two batches.
        The k-th element of the first batch and the k-th element of the second batch are treated as a positive pair.
        """
        # Normalize embeddings
        zi = torch.nn.functional.normalize(zi, dim=-1)  # [batch_size, dim]
        zj = torch.nn.functional.normalize(zj, dim=-1)  # [batch_size, dim]
        
        # Compute similarity matrix between the two batches
        similarity_matrix = torch.mm(zi, zj.t()) / self.temperature  # [batch_size, batch_size]
        
        # Positive pairs are on the diagonal: [i,i] = similarity(zi[i], zj[i])
        positive_sim = torch.diag(similarity_matrix)  # [batch_size]
        
        # For each anchor zi[i], compute denominator: sum over all similarities with zj
        # This includes the positive pair and all negative pairs
        exp_sim = torch.exp(similarity_matrix)  # [batch_size, batch_size]
        denom = exp_sim.sum(dim=1)  # [batch_size]
        
        # Compute NT-Xent loss for each anchor
        loss = -torch.log(torch.exp(positive_sim) / denom)  # [batch_size]
        
        return loss.mean()