import torch
import numpy as np
import matplotlib.pyplot as plt

def mixup_data(x, y, alpha=0.4):
    '''Mix data and labels using MixUp'''
    if alpha > 0:
      lam = np.random.beta(alpha, alpha)
    else:
      lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def show_mixup(dataloader, label_encoder, save_path: str = './results/mixup.png'):
  images, labels = next(iter(dataloader))
  images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
  labels_class = labels['class'].to(images.device)

  from utils.mixup import mixup_data
  mixed_x, y_a, y_b, lam = mixup_data(images, labels_class)

  batch_size = min(4, images.size(0))
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
  inv_label_encoder = {v: k for k, v in label_encoder.items()}

  _, axes = plt.subplots(batch_size, 2, figsize=(10, 4 * batch_size))
  for i in range(batch_size):
      # Original
      orig = images[i] * std + mean
      orig = orig.cpu().permute(1, 2, 0).numpy()
      axes[i, 0].imshow(orig.clip(0, 1))
      axes[i, 0].set_title(f"Original: {inv_label_encoder[int(y_a[i])]}")
      axes[i, 0].set_xticks([])
      axes[i, 0].set_yticks([])

      # MixUp
      mix = mixed_x[i] * std + mean
      mix = mix.cpu().permute(1, 2, 0).numpy()
      axes[i, 1].imshow(mix.clip(0, 1))
      axes[i, 1].set_title(f"MixUp: {inv_label_encoder[int(y_a[i])]} ({lam:.2f}) + {inv_label_encoder[int(y_b[i])]} ({1-lam:.2f})")
      axes[i, 1].set_xticks([])
      axes[i, 1].set_yticks([])
  plt.tight_layout()
  plt.axis('off')
  plt.savefig(save_path)
  plt.show()
