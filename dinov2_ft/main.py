import pandas as pd

import os
print(f'Thư mục hiện tại: {os.getcwd()}')

from transformers import AutoImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader
from dinov2_cls.dataset import ImageTextRetrievalDataset
from dinov2_cls.transform import train_transform
from dinov2_cls.dinov2_bert import contrastive_loss, ImageTextRetrievalModel
import os
import torch

def main(df: pd.DataFrame, num_epochs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = ImageTextRetrievalDataset(df, image_processor, tokenizer, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = ImageTextRetrievalModel("facebook/dinov2-base", "bert-base-uncased").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print('Training start')
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
            img_emb, txt_emb = model(images, encoding['input_ids'], encoding['attention_mask'])
            loss = contrastive_loss(img_emb, txt_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

    save_path = "dinov2_bert_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weight has been saved to {save_path}")

if __name__ == '__main__':
    # import os
    # os.chdir('..')

    json_path = 'Dataset/train/t2i.json'
    img_dir = 'Dataset/train/imgs'

    df = pd.read_json(json_path, typ='series')
    df = df.reset_index()
    df.columns = ['Caption', 'Path']
    df['Path'] = df['Path'].apply(lambda x: os.path.join(img_dir, x))
    print('df has been loaded', df.head())

    main(df, 3)
