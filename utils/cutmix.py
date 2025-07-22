import torch
import numpy as np
import matplotlib.pyplot as plt

def cutmix_data(x, y, alpha=1.0):
    '''Mix data and labels using CutMix'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # Create a random cut box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Choose random center position
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on the area of the cut region
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam

def show_cutmix(dataloader, label_encoder, save_path: str = './results/cutmix.png'):
    images, labels = next(iter(dataloader))
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
    labels_class = labels['class'].to(images.device)

    from utils.cutmix import cutmix_data
    cutmix_x, y_a, y_b, lam = cutmix_data(images, labels_class)

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

        # CutMix
        mix = cutmix_x[i] * std + mean
        mix = mix.cpu().permute(1, 2, 0).numpy()
        axes[i, 1].imshow(mix.clip(0, 1))
        axes[i, 1].set_title(f"CutMix: {inv_label_encoder[int(y_a[i])]} ({lam:.2f}) + {inv_label_encoder[int(y_b[i])]} ({1-lam:.2f})")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()
