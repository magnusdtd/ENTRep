import torch
import torch.nn.functional as F
from CLIP.img_encoder import ImageEncoder
from CLIP.text_encoder import TextEncoder
from CLIP.projection_head import ProjectionHead
from typing import List

class CLIP(torch.nn.Module):
    def __init__(
        self,
        img_embedding_dim: int = 2048,
        img_encoder_unfreeze_layers: List = [],
        text_embedding_dim: int = 768,
        text_encoder_unfreeze_layers: List = [],
        temperature: float = 1.0
    ):
        super().__init__()
        self.img_encoder = ImageEncoder(unfreeze_layers=img_encoder_unfreeze_layers)
        self.text_encoder = TextEncoder(unfreeze_layers=text_encoder_unfreeze_layers)
        self.image_projection = ProjectionHead(embedding_dim=img_embedding_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.img_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        img_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ img_embeddings.T) / self.temperature
        images_similarity = img_embeddings @ img_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
