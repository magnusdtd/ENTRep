import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize

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

def show_mosaic(dataloader, label_encoder, save_path: str = './results/mosaic.png'):

    images, labels = next(iter(dataloader))
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
    labels_class = labels['class'].to(images.device)

    # Take the first 4 images and labels
    images4 = images[:4]
    labels4 = labels_class[:4]

    # Create a batch of 4 (for compatibility with mosaic_data), but only use the first mosaic
    mosaic_imgs, mosaic_labels, lam_list = mosaic_data(images4, labels4)
    mosaic_img = mosaic_imgs[0]  # Only use the first mosaic

    # Prepare for plotting
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    inv_label_encoder = {v: k for k, v in label_encoder.items()}

    # Prepare 2x2 grid of originals with space between images
    orig_imgs = [images4[i] * std + mean for i in range(4)]
    orig_imgs = [img.cpu().permute(1, 2, 0).numpy().clip(0, 1) for img in orig_imgs]
    h, w, _ = orig_imgs[0].shape
    pad = int(0.08 * min(h, w))
    grid_h = 2 * h + pad
    grid_w = 2 * w + pad
    grid = np.ones((grid_h, grid_w, 3))

    # Place images with padding
    grid[0: h, 0: w] = orig_imgs[0]
    grid[0: h, w + pad: w + pad + w] = orig_imgs[1]
    grid[h + pad: h + pad + h, 0: w] = orig_imgs[2]
    grid[h + pad: h + pad + h, w + pad: w + pad + w] = orig_imgs[3]

    # --- Add space between the 2x2 grid and the mosaic image ---
    # We'll create a new image that concatenates the grid and the mosaic image with a visible space between them

    # Prepare mosaic image for display
    mosaic_img_disp = mosaic_img * std + mean
    mosaic_img_disp = mosaic_img_disp.cpu().permute(1, 2, 0).numpy().clip(0, 1)

    # Set the space (in pixels) between the grid and the mosaic image
    space_px = int(min(h, w))

    # Make sure both images have the same height
    grid_disp = grid
    mosaic_disp = mosaic_img_disp
    if grid_disp.shape[0] != mosaic_disp.shape[0]:
        # Resize mosaic to match grid height
        mosaic_disp = resize(mosaic_disp, (grid_disp.shape[0], grid_disp.shape[1]), preserve_range=True, anti_aliasing=True)
        mosaic_disp = np.clip(mosaic_disp, 0, 1)

    # Create the white space
    space = np.ones((grid_disp.shape[0], space_px, 3))

    # Concatenate: [grid | space | mosaic]
    concat_img = np.concatenate([grid_disp, space, mosaic_disp], axis=1)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(concat_img)
    ax.set_xticks([])
    ax.set_yticks([])

    # Titles for the two parts
    label_grid = '\n'.join([f"{inv_label_encoder[int(labels4[i])]}" for i in range(4)])
    label_info = "\n".join([
        f"{inv_label_encoder[int(mosaic_labels[0, k])]} ({lam_list[0, k]:.2f})" for k in range(4)
    ])
    # Place titles above each section
    ax.text(grid_disp.shape[1] // 2, -10, f"Originals images:\n{label_grid}", ha='center', va='bottom', fontsize=12, color='black', transform=ax.transData)
    ax.text(grid_disp.shape[1] + space_px + mosaic_disp.shape[1] // 2, -10, f"Mosaic:\n{label_info}", ha='center', va='bottom', fontsize=12, color='black', transform=ax.transData)

    # --- Draw the arrow in the space between the grid and the mosaic image ---
    # Arrow start at the center of the right edge of the grid, and end at the center of the left edge of the mosaic image

    y_center = grid_disp.shape[0] // 2
    x_start = grid_disp.shape[1] - 1
    x_end = grid_disp.shape[1] + space_px

    # The arrow have the same length of the space
    arrow = mpatches.FancyArrowPatch(
        (x_start + 100, y_center), (x_end - 100, y_center),
        arrowstyle='->',
        mutation_scale=30,
        color='black',
        linewidth=3,
        shrinkA=0, shrinkB=0
    )
    ax.add_patch(arrow)

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
