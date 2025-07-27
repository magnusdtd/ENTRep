import torch.nn as nn
from transformers import AutoModel
import torch

class ImageTextRetrievalModel(nn.Module):
    def __init__(self, image_model_name, text_model_name, embed_dim=512):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image).last_hidden_state[:, 0]  # [CLS] token
        txt_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        img_emb = self.img_proj(img_feat)
        txt_emb = self.txt_proj(txt_feat)
        img_emb = nn.functional.normalize(img_emb, dim=-1)
        txt_emb = nn.functional.normalize(txt_emb, dim=-1)
        return img_emb, txt_emb

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
