# Crowd Infosys – Crowd Counting & Alert System

This project implements a **Crowd Counting and Alert System** using the **CSRNet deep learning model**. It estimates people in a video stream, generates density maps, and sends real-time alerts when the count crosses a threshold.

---

## 📌 Features
- Crowd density estimation with **CSRNet (VGG16 backbone)**
- Real-time visualization in a **Streamlit dashboard**
- Email alert system (using Gmail SMTP + App Passwords)
- Ngrok integration for public deployment

---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/Bhuvanmj/crowd_infosys.git
cd crowd_infosys

2. Install dependencies

If you have a requirements.txt:pip install -r requirements.txt

Or install manually:pip install torch torchvision matplotlib pillow tqdm scipy streamlit opencv-python-headless twilio yagmail pyngrok

3. Model checkpoint3. Model checkpoint
Download or train the CSRNet model and place the file here:/content/drive/MyDrive/ShanghaiTech/best_csrnet.pth

4. Configure Email Alerts
In both crowd.ipynb or crowd.py, update these values:
FROM_EMAIL = "your_email@gmail.com"
TO_EMAIL = "receiver_email@gmail.com"
APP_PASSWORD = "your_google_app_password"

Note: Gmail requires you to generate an App Password. Guide: Google App Passwords

5. Run the app
Using the notebook:

Open crowd.ipynb in Google Colab or Jupyter Notebook

Run the cells in order
Using the script:streamlit run crowd.py --server.port 8501 --server.headless true

Public Deployment with Ngrok:
Add your Ngrok Auth Token:
ngrok authtoken YOUR_NGROK_KEY

Start the Streamlit app:
!nohup streamlit run crowd.py --server.port 8501 --server.headless true &

Expose it with Ngrok:
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("Your app is live at:", public_url)

6. How It Works?

1.Upload a video in the dashboard

2.Model generates density maps + counts

3.Alert is triggered when threshold is crossed

4.Email is sent with the crowd count

7. NOTE:training for multiple epochs increases the accuracy of the output

8. AUTHOR:
Bhuvan MJ
AI & ML Engineer | Passionate about Computer Vision & Deep Learning




