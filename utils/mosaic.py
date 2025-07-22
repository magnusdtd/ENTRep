import torch
import numpy as np

def mosaic_data(x, y):
    '''Mix data and labels using Mosaic'''
    batch_size, c, h, w = x.size()
    mosaic_imgs = torch.zeros_like(x)
    mosaic_labels = torch.zeros((batch_size, 4), dtype=y.dtype, device=x.device)
    lam_list = torch.zeros((batch_size, 4), dtype=torch.float, device=x.device)

    for i in range(batch_size):
        indices = torch.randperm(batch_size)[:4].to(x.device)
        imgs = x[indices]
        labels = y[indices]

        # Mosaic center
        xc = np.random.randint(int(0.25 * w), int(0.75 * w))
        yc = np.random.randint(int(0.25 * h), int(0.75 * h))

        # Coordinates for each image
        x1a, y1a, x2a, y2a = 0, 0, xc, yc
        x1b, y1b, x2b, y2b = xc, 0, w, yc
        x1c, y1c, x2c, y2c = 0, yc, xc, h
        x1d, y1d, x2d, y2d = xc, yc, w, h

        coords = [
            (x1a, y1a, x2a, y2a),
            (x1b, y1b, x2b, y2b),
            (x1c, y1c, x2c, y2c),
            (x1d, y1d, x2d, y2d)
        ]

        mosaic = torch.zeros((c, h, w), device=x.device)
        for k, (x1, y1, x2, y2) in enumerate(coords):
            img = imgs[k]
            img_resized = torch.nn.functional.interpolate(img.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            mosaic[:, y1:y2, x1:x2] = img_resized
            lam_list[i, k] = ((x2 - x1) * (y2 - y1)) / (h * w)
            mosaic_labels[i, k] = labels[k]
        mosaic_imgs[i] = mosaic
    return mosaic_imgs, mosaic_labels, lam_list