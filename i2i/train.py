import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize, softmax
import copy
import math
import torch
import copy
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from prototype_clf.ENTRepDataset import ENTRepDataset

class ProjectionModel(torch.nn.Module):
    def __init__(
            self,
            input_dim, 
            embedder_dims, 
            projection_dim=512, 
            use_layernorm=False, 
            use_dropout=False, 
            dropout_rate=0.1, 
            internal_dim=1024, 
            use_attention=False, 
            attention_dim=512, 
            extra_layer=False
        ):
        super().__init__()
        self.use_attention = use_attention
        self.num_embedders = len(embedder_dims) if use_attention else None
        self.embedder_dims = embedder_dims if use_attention else None
        self.fixed_dim = attention_dim

        if use_attention:
            self.attn_weights = torch.nn.Parameter(torch.ones(self.num_embedders, dtype=torch.float32))

            self.attn_projections = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, self.fixed_dim), torch.nn.ReLU()
                ) for emb_dim in embedder_dims
            ])
        
        layers = [
            torch.nn.Linear(self.fixed_dim if use_attention else input_dim, internal_dim * 2 if extra_layer else internal_dim),
            torch.nn.ReLU()
        ]

        if use_layernorm:
            layers.append(torch.nn.LayerNorm(internal_dim))
        if use_dropout:
            layers.append(torch.nn.Dropout(dropout_rate))

        if extra_layer:
            layers.extend([torch.nn.Linear(internal_dim * 2, internal_dim), torch.nn.ReLU()])

        layers.append(torch.nn.Linear(internal_dim, projection_dim))

        if use_layernorm:
            layers.append(torch.nn.LayerNorm(projection_dim))

        self.projection = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_attention:
            if x.dim() == 3 and x.shape[1] == 1:
                x = x.squeeze(1)  # Squeeze out the second dimension
            # split the embeddings out
            start = 0
            embeddings = []
            for emb_dim, proj_layer in zip(self.embedder_dims, self.attn_projections):
                embedding = x[:, start:start + emb_dim]
                projected_embedding = proj_layer(embedding)
                embeddings.append(projected_embedding)
                start += emb_dim
            weights = softmax(self.attn_weights, dim=0)
            for embed in embeddings:
                print(f'embed.shape = {embed.shape}')
            x = torch.stack([w * e for w, e in zip(weights, embeddings)], dim=0).sum(dim=0)
            print("after attention projections:", x.shape)
            x = self.projection(x)
            print("after final projection:", x.shape)
        else:
            x = self.projection(normalize(x, p=2, dim=-1))

        return normalize(x, p=2, dim=-1)

def train(
    model: ProjectionModel, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs=300, 
    patience=5, 
    lr=1e-5, 
    device='cuda',
    lr_factor = 0.5
):
    early_stopper = EarlyStopping(patience=patience, mode='min', metric_name='Val Loss')
    
    model.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()), 
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=2, verbose=True)
    infonce_loss_func = NTXentLoss(temperature=0.07).to(device)
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for _, _, _, x in train_iter:
            x = x.to(device)
            embeddings = model(x)
            infonce_loss = infonce_loss_func(embeddings, torch.arange(embeddings.size(0), device=device))
            loss = infonce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_iter.set_postfix({
                "InfoNCE": f"{total_loss:.3f}"
            })

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
            for _, _, _, x in val_iter:
                x = x.to(device)
                embeddings = model(x)
                infonce_loss = infonce_loss_func(embeddings, torch.arange(embeddings.size(0), device=device))
                val_loss += infonce_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1:3d} | "
            f"InfoNCE: {total_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        scheduler.step(avg_val_loss)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        # Early stopping
        early_stopper(model, avg_val_loss)
        if early_stopper.early_stop:
            break
        if early_stopper.best_model_state is not None:
            best_model_state = copy.deepcopy(early_stopper.best_model_state)
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model
    