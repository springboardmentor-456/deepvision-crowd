import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import smtplib
from email.message import EmailMessage
class CrowdCNN(nn.Module):
    def __init__(self):
        super(CrowdCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    def forward(self, x):
        return self.features(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
EMAIL_ADDRESS = "@gmail.com"
EMAIL_PASSWORD = "password"
crowd_threshold = 200
def send_email_alert(count):
    msg = EmailMessage()
    msg['Subject'] = " Overcrowding Alert!"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = "@gmail.com"
    msg.set_content(f"Alert! Crowd count {int(count)} exceeds threshold.")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
def preprocess_image(image):
    img = np.array(image)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img, (512, 512))
    img_resized = img_resized / 255.0
    img_resized = (img_resized - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    img_resized = np.transpose(img_resized, (2,0,1))
    tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
    return tensor.to(device), img
st.set_page_config(page_title="DeepVision Crowd Monitor", layout="wide")
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
trigger = st.sidebar.button(" Estimate Crowd")
st.title("📊 DeepVision Crowd Monitor")
st.write("Upload an image to estimate crowd count and visualize density map.")
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False
if uploaded_file is not None and trigger:
    image = Image.open(uploaded_file)
    input_tensor, original_image = preprocess_image(image)
    with torch.no_grad():
        density_map = model(input_tensor)
    density_map_np = density_map.squeeze().cpu().numpy()
    density_map_np = np.clip(density_map_np, 0, None)
    crowd_count = np.sum(density_map_np)
    density_map_resized = cv2.resize(density_map_np, (original_image.shape[1], original_image.shape[0]))
    density_map_colored = cv2.applyColorMap((density_map_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, density_map_colored, 0.4, 0)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(original_image, channels="RGB")
    with col2:
        st.subheader("Crowd Density Map")
        st.image(overlay, channels="RGB")
    st.metric(label="Estimated Crowd Count", value=int(crowd_count))
    if crowd_count > crowd_threshold:
        st.error(f"⚠️ Crowd exceeds threshold ({crowd_threshold})!")
        if not st.session_state.alert_sent:
            send_email_alert(crowd_count)
            st.success("✅ Alert email sent!")
            st.session_state.alert_sent = True
