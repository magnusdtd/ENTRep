import torch
from tqdm import tqdm
import torch.nn.functional as F

class Evaluator:
    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _extract_embeddings(self):
        self.model.eval()
        all_img_embeds = []
        all_txt_embeds = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Extracting embeddings"):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                img_embeds = self.model.image_projection(self.model.img_encoder(batch["image"]))
                txt_embeds = self.model.text_projection(
                    self.model.text_encoder(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    )
                )
                all_img_embeds.append(F.normalize(img_embeds, dim=-1))
                all_txt_embeds.append(F.normalize(txt_embeds, dim=-1))
        all_img_embeds = torch.cat(all_img_embeds, dim=0)
        all_txt_embeds = torch.cat(all_txt_embeds, dim=0)
        return all_img_embeds, all_txt_embeds

    def recall_at_k(self, k=1):
        """
        Computes recall@k for both text-to-image and image-to-text retrieval.
        """
        img_embeds, txt_embeds = self._extract_embeddings()
        sim_matrix = txt_embeds @ img_embeds.T
        n = sim_matrix.size(0)

        # T2I top k
        t2i_topk = sim_matrix.topk(k, dim=1).indices
        t2i_recall = (
            torch.arange(n).unsqueeze(1).to(t2i_topk.device) == t2i_topk
        ).any(dim=1).float().mean().item()

        # I2T top k
        i2t_topk = sim_matrix.topk(k, dim=0).indices
        i2t_recall = (
            torch.arange(n).unsqueeze(1).to(i2t_topk.device) == i2t_topk
        ).any(dim=1).float().mean().item()

        return {
            f"text_to_image_recall@{k}": t2i_recall, 
            f"image_to_text_recall@{k}": i2t_recall
        }

    def mean_reciprocal_rank(self):
        """
        Computes mean reciprocal rank (MRR) for both text-to-image and image-to-text retrieval.
        """
        img_embeds, txt_embeds = self._extract_embeddings()
        sim_matrix = txt_embeds @ img_embeds.T
        n = sim_matrix.size(0)

        # T2I MRR
        t2i_ranks = sim_matrix.argsort(dim=1, descending=True)
        t2i_rr = 1.0 / ( (t2i_ranks == torch.arange(n).unsqueeze(1)).nonzero()[:,1].float() + 1 )
        t2i_mrr = t2i_rr.mean().item()
        
        # I2T MRR
        i2t_ranks = sim_matrix.argsort(dim=0, descending=True)
        i2t_rr = 1.0 / ( (i2t_ranks == torch.arange(n).unsqueeze(1)).nonzero()[:,1].float() + 1 )
        i2t_mrr = i2t_rr.mean().item()

        return {
            "text_to_image_mrr": t2i_mrr, 
            "image_to_text_mrr": i2t_mrr
        }