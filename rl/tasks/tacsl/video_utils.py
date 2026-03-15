

import torch
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

def tensor_to_pil(img_tensor):
    if img_tensor.dtype == torch.float32:
        img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
    
    if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:  # (C, H, W)
        img_tensor = img_tensor.permute(1, 2, 0)  # (H, W, C)
    elif img_tensor.dim() == 2:  # grayscale (H, W)
        img_tensor = img_tensor.unsqueeze(-1)

    img_np = img_tensor.squeeze(0).cpu().numpy()
    
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)  # grayscale → RGB

    return Image.fromarray(img_np)


def resize_to(img, target_height):
    w, h = img.size
    new_w = int(w * target_height / h)
    return img.resize((new_w, target_height))


def make_frame_from_obs(images):
    h = 500  # common height for stacking

    keys = list(images.keys())
    shear = False
    # check if any of the keys include 'shear'
    if any('shear' in key for key in keys):
        shear = True
        

    if not shear:
        row1 = [resize_to(tensor_to_pil(images[k]), h) for k in ['front', 'wrist']]
        row2 = [resize_to(tensor_to_pil(images[k]), h) for k in ['side', 'left_tactile_camera_taxim', 'right_tactile_camera_taxim']]
        # row2 = [resize_to(tensor_to_pil(images[k]), h) for k in ['left_tactile_camera', 'right_tactile_camera']]
    else:
        row1 = [resize_to(tensor_to_pil(images[k]), h) for k in ['front', 'wrist', 'value_plot']]
        row2_side = [resize_to(tensor_to_pil(images[k]), h) for k in ['side']]
        row2_shear = [resize_to(tensor_to_pil(images[k].to(torch.float32).permute(1,0,2)), h) for k in ['tactile_force_field_left_shear', 'tactile_force_field_right_shear']]
        row2 = row2_side + row2_shear

    def hstack_row(imgs):
        total_w = sum(i.width for i in imgs)
        row = Image.new('RGB', (total_w, h))
        x = 0
        for img in imgs:
            row.paste(img, (x, 0))
            x += img.width
        return row

    frame = Image.new('RGB', (max(r.width for r in [*map(hstack_row, [row1, row2])]), h * 2))
    for i, row in enumerate([row1, row2]):
        frame.paste(hstack_row(row), (0, h * i))

    return frame


def value_plot_to_numpy(pred_value_history):
    fig, ax = plt.subplots(figsize=(8, 8))  # small figure
    ax.plot(pred_value_history, color='blue')
    ax.set_title("Predicted Value")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_xlim(0, 250)
    ax.set_ylim(-1.5, 1)
    ax.grid(True)
    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_np = np.array(image)
    image_torch = torch.from_numpy(image_np)
    
    return image_torch