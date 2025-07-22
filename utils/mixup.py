import torch
import numpy as np

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
