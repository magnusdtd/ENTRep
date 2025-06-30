import torch
from torch.nn.functional import normalize
import numpy as np

class PrototypeClassifier(torch.nn.Module):
    def __init__(self, train_dataset, projection_model, device='cuda'):
        super().__init__()
        self.device = device
        self.train_dataset = train_dataset
        self.projection_model = projection_model.to(self.device)
        self.projection_model.eval()

        class_embeddings, _ = self._get_classifier_embeddings(train_dataset)
        self.class_embeddings = [self.projection_model(class_embedding.to(device)) for class_embedding in class_embeddings]
        self.class_prototypes = torch.nn.Parameter(self.get_mean_prototypes(self.class_embeddings), requires_grad=False)


    def _get_classifier_embeddings(self, train_dataset):
        class_embeddings = []
        empty_classes = []
        n_classes = min(torch.inf, train_dataset.n_classes)
        for cl in range(n_classes):
            cls_embs = train_dataset.get_embeddings_for_class(cl)
            if len(cls_embs) == 0:
                empty_classes.append(cl)
                class_embeddings.append(torch.zeros(1, train_dataset.emb_dim, dtype=torch.float32))
            else:
                class_embeddings.append(torch.tensor(np.vstack(cls_embs.values), dtype=torch.float32))
        return class_embeddings, empty_classes

    def get_mean_prototypes(self, embeddings):
        return torch.stack([class_embs.mean(dim=0) for class_embs in embeddings]).float()

    @torch.no_grad()
    def make_prediction(self, embeddings):
        embeddings = embeddings.to(self.device).float()
        embeddings = self.projection_model(embeddings)
        print(embeddings.shape)
        print(self.class_prototypes.shape)
        
        if embeddings.dim() == 2 and embeddings.shape[1] != 1:
            embeddings = embeddings.unsqueeze(1)

        embeddings = normalize(embeddings, p=2, dim=-1)

        similarities = torch.nn.functional.cosine_similarity(embeddings, self.class_prototypes, dim=-1)
        probas = torch.nn.functional.softmax(similarities, dim=1).detach().cpu()

        return probas
