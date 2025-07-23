import torch
from transformers import DistilBertModel, DistilBertConfig
from typing import List
from utils.unfreeze_layer import unfreeze_model_layers

class TextEncoder(torch.nn.Module):
    def __init__(
        self, 
        model_name: str = 'distilbert-base-uncased', 
        pretrained: bool = False, 
        unfreeze_layers: List = []
    ):
        super().__init__()
        
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        if unfreeze_layers:
            unfreeze_model_layers(self.model, unfreeze_layers)

        # Using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

