import torch

class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        dropout_ratio: int = 0.4
    ):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.norm_layer = torch.nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.dropout(self.fc(self.gelu(projected)))
        x += projected
        return self.norm_layer(x)
