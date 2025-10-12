import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from PIL import Image

# Dataset Class with Adaptive Gaussian

class ShanghaiTechDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None, use_adaptive=True, fixed_sigma=15):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.use_adaptive = use_adaptive
        self.fixed_sigma = fixed_sigma

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # load ground-truth points
        gt_name = f"GT_{os.path.splitext(img_name)[0]}.mat"
        gt_path = os.path.join(self.gt_dir, gt_name)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0, 0][0, 0][0]

        # density map initialization
        density_map = np.zeros((h, w), dtype=np.float32)

        if len(points) > 0:
            # use adaptive sigma
            if self.use_adaptive and len(points) > 3:
                tree = KDTree(points.copy(), leafsize=2048)
                for i, p in enumerate(points):
                    x, y = min(int(p[0]), w - 1), min(int(p[1]), h - 1)
                    sigma = np.mean(tree.query(p, k=4)[0][1:]) * 0.1
                    sigma = max(1, sigma)  # ensure sigma >=1
                    density_map[y, x] = 1
                    density_map = gaussian_filter(density_map, sigma=sigma)
            else:
                # fixed sigma fallback
                for p in points:
                    x, y = min(int(p[0]), w - 1), min(int(p[1]), h - 1)
                    density_map[y, x] = 1
                density_map = gaussian_filter(density_map, sigma=self.fixed_sigma)

        if self.transform:
            img = self.transform(Image.fromarray(img))
            density_map = cv2.resize(density_map, (img.shape[2], img.shape[1]))
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        count = density_map.sum().item()
        return img, density_map, count, img_name

# ==============================
# Visualization Function
# ==============================
def visualize_sample(dataset, idx=0, save_path="visualizations/sample.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img, gt, count, name = dataset[idx]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Image: {name}")
    plt.subplot(1, 2, 2)
    plt.imshow(gt.squeeze(0), cmap="jet")
    plt.title(f"Density Map\nCount: {count:.1f}")
    plt.colorbar()
    plt.savefig(save_path)
    plt.show()

# ==============================
# Main Program
# ==============================
if __name__ == "__main__":
    image_dir = r"/content/drive/MyDrive/Shang_data/ShanghaiTech/part_A/train_data/images"
    gt_dir = r"/content/drive/MyDrive/Shang_data/ShanghaiTech/part_A/train_data/ground-truth"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])

    dataset = ShanghaiTechDataset(image_dir=image_dir, gt_dir=gt_dir, transform=transform, use_adaptive=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Total samples: {len(dataset)}")

    # visualize 1st sample
    visualize_sample(dataset, idx=7)
