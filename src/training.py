import os, glob, math
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=load_weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=2,dilation=2), nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64,1,1),
            nn.ReLU(inplace=True)  
        )

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

def make_density(img, points, sigma=4):
    h, w = img.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    if points is None or points.size == 0:
        return density
    points = np.array(points, dtype=np.float32)
    if points.ndim == 0:
        return density
    if points.ndim == 1:
        if points.shape[0] == 2:
            points = points.reshape(1,2)
        else:
            return density
    for x, y in points:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            density[y, x] = 1
    density = gaussian_filter(density, sigma=sigma)
    return density

class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None, sigma=4):
        self.imgs = sorted(glob.glob(os.path.join(img_dir,"*.jpg")) +
                           glob.glob(os.path.join(img_dir,"*.png")))
        self.gt_dir = gt_dir
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,i):
        img_path = self.imgs[i]
        img = Image.open(img_path).convert("RGB")
        base_noext = os.path.splitext(os.path.basename(img_path))[0]

        mat_path = os.path.join(self.gt_dir, f"GT_{base_noext}.mat")
        pts = np.zeros((0,2), dtype=np.float32)

        try:
            mat = loadmat(mat_path)
            info = mat['image_info'][0,0]
            if isinstance(info, np.ndarray) and info.size == 1:
                info = info[0,0]
            if 'location' in info.dtype.names:
                loc = info['location']
                if loc.size > 0:
                    while isinstance(loc, np.ndarray) and loc.size == 1:
                        loc = loc[0,0] if loc.ndim==2 else loc
                    pts = np.array(loc, dtype=np.float32)
                    if pts.ndim==1 and pts.shape[0]==2:
                        pts = pts.reshape(1,2)
        except Exception as e:
            print(f"[WARNING] Failed to parse {mat_path}: {e}")
            pts = np.zeros((0,2), dtype=np.float32)

        den = make_density(np.array(img), pts, sigma=self.sigma)
        num_points = pts.shape[0] if pts.ndim>0 else 0
        print(f"[DEBUG] Image: {base_noext}, Points: {num_points}, GT Density sum: {den.sum():.2f}")

        if self.transform:
            img = self.transform(img)
        den = torch.from_numpy(den).unsqueeze(0).float()
        return img, den

def train_model(train_loader, val_loader, num_epochs=50, lr=1e-5, save_path="./csrnet_partB_full.pth"):
    model = CSRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0
        for imgs, dens in train_loader:
            imgs, dens = imgs.to(device), dens.to(device)
            preds = model(imgs)
            dens_resized = nn.functional.interpolate(dens, size=(preds.shape[2], preds.shape[3]),
                                                     mode='bilinear', align_corners=False)
            factor_h = preds.shape[2]/dens.shape[2]
            factor_w = preds.shape[3]/dens.shape[3]
            dens_resized = dens_resized / (factor_h * factor_w)

            loss = loss_fn(preds, dens_resized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        total_mae, total_mse = 0,0
        with torch.no_grad():
            for imgs, dens in val_loader:
                imgs, dens = imgs.to(device), dens.to(device)
                preds = model(imgs)
                dens_resized = nn.functional.interpolate(dens, size=(preds.shape[2], preds.shape[3]),
                                                         mode='bilinear', align_corners=False)
                factor_h = preds.shape[2]/dens.shape[2]
                factor_w = preds.shape[3]/dens.shape[3]
                dens_resized = dens_resized / (factor_h * factor_w)
                total_mae += abs(preds.sum() - dens.sum()).item()
                total_mse += (preds.sum() - dens.sum()).item()**2

        val_mae = total_mae / len(val_loader)
        val_rmse = math.sqrt(total_mse / len(val_loader))
        print(f"Epoch {epoch}/{num_epochs}: Train Loss={avg_loss:.6f}, Val MAE={val_mae:.2f}, Val RMSE={val_rmse:.2f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Checkpoint saved at epoch {epoch}")

    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Training finished. Model saved at {save_path}")
    return model

if __name__=="__main__":
    img_dir = "/content/drive/MyDrive/ColabDataset/ShanghaiTech/Data/part_B/train_data/images" #replace with images path in the ShanghaiTech dataset
    gt_dir  = "/content/drive/MyDrive/ColabDataset/ShanghaiTech/Data/part_B/train_data/ground-truth"  #replace with ground-truth path in the ShanghaiTech dataset

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = CrowdDataset(img_dir, gt_dir, transform=transform, sigma=8)

    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = train_model(train_loader, val_loader, num_epochs=50, lr=1e-5, save_path="./csrnet_train.pth")
