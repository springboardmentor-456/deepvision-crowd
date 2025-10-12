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

def load_model(model_path=None):
    """Load CSRNet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSRNet()
    model_loaded = False

    if model_path and os.path.exists(model_path):
        try:
            # Clear cache first
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)

            # Verify model loaded correctly
            sample_input = torch.randn(1, 3, 512, 512).to(device)
            model.to(device)
            model.eval()
            with torch.no_grad():
                test_output = model(sample_input)

            st.success(f"✅ Loaded trained model from {os.path.basename(model_path)}")
            st.info(f"Model output shape: {test_output.shape}")
            model_loaded = True

        except Exception as e:
            st.error(f"❌ Could not load model weights: {e}")
            st.warning("Using untrained model - results will be meaningless!")
            import traceback
            st.code(traceback.format_exc())
    else:
        if model_path:
            st.error(f"⚠️ **Model file not found:** {model_path}")
        else:
            st.error("⚠️ **CRITICAL: No pre-trained model provided!**")
        st.warning("The model is using random weights and will produce incorrect results.")
        st.info("Please provide a valid path to trained weights.")

    model.to(device)
    model.eval()
    return model, device, model_loaded

# --------------------------
# Predict crowd density
# --------------------------
def predict_density(model, image, device):
    """Generate density map prediction from model"""
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        density_map = model(image_tensor)
        density_map = density_map.squeeze(0).squeeze(0).cpu().numpy()
    return density_map

# --------------------------
# Density map generation & visualization
# --------------------------
def generate_density_map(points, image_shape, sigma=15):
    """Generate ground truth density map from point annotations"""
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            k = int(6 * sigma + 1)
            if k % 2 == 0:
                k += 1
            xx, yy = np.meshgrid(np.arange(k), np.arange(k))
            c = k // 2
            g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2))
            g /= (2 * np.pi * sigma ** 2)
            x_min, x_max = max(0, x - c), min(w, x + c + 1)
            y_min, y_max = max(0, y - c), min(h, y + c + 1)
            kx_min, kx_max = max(0, c - x), max(0, c - x) + x_max - x_min
            ky_min, ky_max = max(0, c - y), max(0, c - y) + y_max - y_min
            density_map[y_min:y_max, x_min:x_max] += g[ky_min:ky_max, kx_min:kx_max]
    return density_map

def visualize_density_map(image_tensor, density_map, estimated_count, original_image=None):
    """Visualize the density map and overlay"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image')
    axes[0].axis('off')

    # Density map
    im = axes[1].imshow(density_map, cmap='jet')
    axes[1].set_title(f'Density Map\nEstimated Count: {estimated_count:.0f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(image, alpha=0.7)
    axes[2].imshow(density_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()