import torch

class LearnableLossWeighting(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.log_sigma_ce = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma_triplet = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, ce_loss, triplet_loss):
        # From Kendall et al. CVPR 2018
        loss = (
            torch.exp(-self.log_sigma_ce) * ce_loss +
            torch.exp(-self.log_sigma_triplet) * triplet_loss +
            self.log_sigma_ce + self.log_sigma_triplet
        )
        return 0.5 * loss
