%%writefile /content/streamlit_app.py


import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import smtplib
from email.mime.text import MIMEText
import os

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    st.warning("⚠️ Twilio not installed. Run: !pip install twilio")

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=load_weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

def send_alert_email(recipient_email, current_count, threshold):
    """Sends an email alert when crowd threshold is exceeded."""
    try:
        sender_email = os.environ.get('SENDER_EMAIL')
        sender_password = os.environ.get('SENDER_PASSWORD')

        if not sender_email or not sender_password:
            st.error("❌ Email credentials not set in environment variables.")
            return False

        if not recipient_email or '@' not in recipient_email:
            st.error("❌ Invalid recipient email address.")
            return False

        subject = "🚨 Crowd Alert: Threshold Exceeded!"
        body = f"""
CROWD ALERT NOTIFICATION
========================

⚠️ The crowd count has exceeded the safety threshold!

Current Count: {current_count:.2f} people
Threshold Set: {threshold} people
Exceeded By: {current_count - threshold:.2f} people

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Please take necessary action immediately.

---
This is an automated alert from the Crowd Counting System.
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        st.success(f"✅ Email sent to {recipient_email}")
        return True

    except Exception as e:
        st.error(f"❌ Email failed: {str(e)}")
        return False

def send_sms_alert(recipient_phone, current_count, threshold):
    """Sends an SMS alert via Twilio when crowd threshold is exceeded."""
    if not TWILIO_AVAILABLE:
        st.error("❌ Twilio library not installed. Run: !pip install twilio")
        return False

    try:
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        twilio_phone = os.environ.get('TWILIO_PHONE_NUMBER')

        if not all([account_sid, auth_token, twilio_phone]):
            st.error("❌ Twilio credentials not set.")
            st.info("Set: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
            return False

        if not recipient_phone.startswith('+'):
            st.error("❌ Phone must be in E.164 format (e.g., +1234567890)")
            return False

        client = Client(account_sid, auth_token)

        message_body = f"""🚨 CROWD ALERT!

Current: {current_count:.0f} people
Threshold: {threshold}
Exceeded by: {current_count - threshold:.0f}

Time: {time.strftime('%H:%M:%S')}

Take immediate action!"""

        message = client.messages.create(
            body=message_body,
            from_=twilio_phone,
            to=recipient_phone
        )

        st.success(f"✅ SMS sent to {recipient_phone}")
        st.info(f"📱 Message SID: {message.sid}")
        return True

    except Exception as e:
        st.error(f"❌ SMS failed: {str(e)}")
        return False


def send_combined_alerts(recipient_email, recipient_phone, current_count, threshold,
                        send_email=True, send_sms=True):
    """Sends both email and SMS alerts."""
    results = {'email_sent': False, 'sms_sent': False}

    if send_email and recipient_email:
        results['email_sent'] = send_alert_email(recipient_email, current_count, threshold)

    if send_sms and recipient_phone:
        results['sms_sent'] = send_sms_alert(recipient_phone, current_count, threshold)

    return results


st.set_page_config(page_title="Crowd Counting Dashboard", layout="wide")
st.title("🟢 Crowd Counting Dashboard with Smart Alerts")


st.sidebar.header("⚙️ Settings")


threshold = st.sidebar.slider("Crowd Threshold", min_value=10, max_value=500, value=50, step=5)
frame_skip = st.sidebar.number_input("Frame Interval (process every Nth frame)", min_value=1, value=5)

st.sidebar.divider()

st.sidebar.subheader("📬 Alert Configuration")

enable_email = st.sidebar.checkbox("Enable Email Alerts", value=True)
recipient_email = ""
if enable_email:
    recipient_email = st.sidebar.text_input("Recipient Email", value="", placeholder="user@example.com")

enable_sms = st.sidebar.checkbox("Enable SMS Alerts", value=False)
recipient_phone = ""
if enable_sms:
    recipient_phone = st.sidebar.text_input(
        "Recipient Phone",
        value="",
        placeholder="+1234567890",
        help="E.164 format: +[country code][number]"
    )

st.sidebar.divider()


st.sidebar.subheader("🧪 Test Alerts")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📧 Test Email"):
        if recipient_email:
            with st.spinner("Sending test email..."):
                send_alert_email(recipient_email, 100, 50)
        else:
            st.sidebar.warning("Enter email address first")

with col2:
    if st.button("📱 Test SMS"):
        if recipient_phone:
            with st.spinner("Sending test SMS..."):
                send_sms_alert(recipient_phone, 100, 50)
        else:
            st.sidebar.warning("Enter phone number first")


with st.sidebar.expander("📖 Setup Instructions"):
    st.markdown("""
    **Email Setup (Gmail):**
    1. Get App Password: https://myaccount.google.com/apppasswords
    2. Set environment variables:
       ```python
       os.environ['SENDER_EMAIL'] = "your@gmail.com"
       os.environ['SENDER_PASSWORD'] = "app-password"
       ```

    **SMS Setup (Twilio):**
    1. Sign up at https://www.twilio.com/try-twilio
    2. Get your credentials from Twilio Console
    3. Set environment variables:
       ```python
       os.environ['TWILIO_ACCOUNT_SID'] = "ACxxxxx"
       os.environ['TWILIO_AUTH_TOKEN'] = "your-token"
       os.environ['TWILIO_PHONE_NUMBER'] = "+1234567890"
       ```
    4. Install Twilio: `!pip install twilio`

    **Phone Number Format:**
    - US: +1234567890
    - UK: +441234567890
    - India: +911234567890
    """)


st.sidebar.divider()
video_file = st.file_uploader("📹 Upload Video (mp4)", type=["mp4"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/content/drive/MyDrive/ShanghaiTech/best_csrnet.pth"

try:
    model = CSRNet(load_weights=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


frame_display = st.empty()
count_display = st.empty()
alert_display = st.empty()


if video_file is not None:
    tfile = "./temp_video.mp4"
    with open(tfile, 'wb') as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(tfile)
    frame_count = 0
    crowd_counts = []
    peak_count = 0
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue


        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img_pil).unsqueeze(0).to(device)


        with torch.no_grad():
            output = model(img_t)
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
            density_map = output[0,0].cpu().numpy()
            count = density_map.sum()


        crowd_counts.append(count)
        peak_count = max(peak_count, count)
        avg_count = np.mean(crowd_counts)


        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title("Original Frame")
        axes[0].axis('off')

        axes[1].imshow(density_map, cmap='hot')
        axes[1].set_title(f"Predicted Count = {count:.2f}")
        axes[1].axis('off')

        frame_display.pyplot(fig)
        count_display.metric(
            "Current Crowd Count",
            f"{count:.2f}",
            delta=f"Peak: {peak_count:.2f} | Avg: {avg_count:.2f}"
        )


        if count >= threshold:
            alert_display.warning(f"⚠️ Overcrowding Detected! Count = {count:.2f}")

            if not alert_sent:
                results = send_combined_alerts(
                    recipient_email,
                    recipient_phone,
                    count,
                    threshold,
                    send_email=enable_email,
                    send_sms=enable_sms
                )

                if results['email_sent'] or results['sms_sent']:
                    alert_sent = True
        else:
            alert_display.success("✅ Crowd below threshold")
            alert_sent = False

        plt.close(fig)
        time.sleep(0.1)

    cap.release()
    st.success("🎬 Video processing completed!")