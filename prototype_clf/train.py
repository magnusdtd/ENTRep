import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize, softmax
import copy
import math
import torch
import copy
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from utils.learnable_loss import LearnableLossWeighting
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

class CosineClassifier(torch.nn.Module):
    """Classifier to train the ProjectionModel"""
    def __init__(self, embed_dim, num_classes, scale=10.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_classes, embed_dim, dtype=torch.float32))
        self.scale = scale  # Optional learnable scaling
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: [B, D]
        x = normalize(x, p=2, dim=-1)
        w = normalize(self.weight, p=2, dim=-1)
        return self.scale * torch.matmul(x, w.T)

def train(
    model: ProjectionModel, 
    classifier: CosineClassifier, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs=300, 
    patience=5, 
    lr=1e-5, 
    device='cuda',
    lambda_triplet=None
):
    early_stopper = EarlyStopping(patience=patience, mode='min', metric_name='Val Loss')
    
    model.to(device)
    classifier.to(device)
    if lambda_triplet == "learned" or lambda_triplet is None:
        loss_weighter = LearnableLossWeighting().to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) +
            list(classifier.parameters()) +
            list(loss_weighter.parameters()), 
            lr=lr, weight_decay=1e-4
        )
    elif isinstance(lambda_triplet, float) or isinstance(lambda_triplet, int):
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()), 
            lr=lr, 
            weight_decay=1e-4
        )

    infonce_loss_func = NTXentLoss(temperature=0.07).to(device)

    best_model_state = None
    best_classifier_state = None

    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        total_loss = 0.0
        ce_loss_total = 0.0
        triplet_loss_total = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)

            embeddings = model(x)
            logits = classifier(embeddings)

            ce_loss = F.cross_entropy(logits, y)
            infonce_loss = infonce_loss_func(embeddings, y)

            if lambda_triplet == "learned" or lambda_triplet is None:
                loss = loss_weighter(ce_loss, infonce_loss)
            elif isinstance(lambda_triplet, float) or isinstance(lambda_triplet, int):
                loss = ce_loss + lambda_triplet * infonce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ce_loss_total += ce_loss.item()
            triplet_loss_total += infonce_loss.item()

            train_iter.set_postfix({
                "CE": f"{ce_loss_total:.3f}",
                "InfoNCE": f"{triplet_loss_total:.3f}",
                "Total": f"{total_loss:.3f}"
            })

        # Validation
        model.eval()
        classifier.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
            for x, y in val_iter:
                x, y = x.to(device), y.to(device)
                embeddings = model(x)
                logits = classifier(embeddings)
                ce_loss = F.cross_entropy(logits, y)
                infonce_loss = infonce_loss_func(embeddings, y)

                if lambda_triplet == "learned" or lambda_triplet is None:
                    val_loss += (loss_weighter(ce_loss, infonce_loss)).item()
                elif isinstance(lambda_triplet, float) or isinstance(lambda_triplet, int):
                    val_loss += (ce_loss + lambda_triplet * infonce_loss).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train CE: {ce_loss_total:.4f} | "
            f"InfoNCE: {triplet_loss_total:.4f} | "
            f"Total: {total_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        early_stopper(model, avg_val_loss)
        if early_stopper.early_stop:
            break
        if early_stopper.best_model_state is not None:
            best_model_state = copy.deepcopy(early_stopper.best_model_state)
            best_classifier_state = copy.deepcopy(classifier.state_dict())

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_classifier_state is not None:
        classifier.load_state_dict(best_classifier_state)
    return model, classifier
    