import torch
import math
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from training import CSRNet  
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_csrnet_model(pth_path="../csrnet_model/csrnet_train.pth"):
    model = CSRNet(load_weights=False).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    return model

def get_count_and_heatmap(model, image: Image.Image, scale_small_crowds=True):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    density_map = pred.squeeze().cpu().numpy()
    density_map = np.clip(density_map, 0, None)
    density_map[density_map < 0.01] = 0  # remove very small noise

    # Estimated count
    est_count = float(density_map.sum())

    # Visualization for debugging
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title(f"Raw Density Map — Estimated Count: {est_count:.2f}")
    plt.show()

    # Adjust count for sparse scenes
    if est_count <= 5:  # very sparse crowd
        est_count *= 1.0  # scale up
    elif est_count <= 15:  # medium sparse crowd
        est_count *= 1.3

    # Round up to integer and ensure at least 1
    est_count_final = max(math.ceil(est_count), 1)

    # Create heatmap overlay
    heatmap = density_map / (density_map.max() + 1e-8)
    heatmap_threshold = 0.08 if est_count_final <= 15 else 0.05
    heatmap[heatmap < heatmap_threshold] = 0
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, image.size)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay_rgb, est_count_final

