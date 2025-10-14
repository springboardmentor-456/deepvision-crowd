# Crowd Infosys – Crowd Counting & Alert System

This project implements a Crowd Counting and Alert System using the CSRNet deep learning model.
It estimates the number of people in a video stream, generates density maps, and sends real-time alerts
when the crowd count crosses a defined threshold.

---

## Features

- Crowd density estimation with CSRNet (VGG16 backbone)
- Real-time visualization in a Streamlit dashboard
- Email alert system using Gmail SMTP and App Passwords
- Ngrok integration for public deployment

---

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/Bhuvanmj/crowd_infosys.git  
cd crowd_infosys

### 2. Install Dependencies
If you have a requirements.txt file:  
pip install -r requirements.txt

Or install manually:  
pip install torch torchvision matplotlib pillow tqdm scipy streamlit opencv-python-headless twilio yagmail pyngrok

### 3. Model Checkpoint
Download or train the CSRNet model and place the file here:  
/content/drive/MyDrive/ShanghaiTech/best_csrnet.pth

### 4. Configure Email Alerts
In both DEEP_VISION_CROWD_MONITERING.ipynb or crowd.py, update these values:

FROM_EMAIL = your_email@gmail.com  
TO_EMAIL = receiver_email@gmail.com  
APP_PASSWORD = your_google_app_password

Note: Gmail requires you to generate an App Password if 2-Step Verification is enabled.

### 5. Run the Application

**Using the Notebook:**  
Open DEEP_VISION_CROWD_MONITERING.ipynb in Google Colab or Jupyter Notebook.  
Run all the cells in order.

**Using the Script:**  
streamlit run crowd.py --server.port 8501 --server.headless true

### 6. Public Deployment with Ngrok

Add your Ngrok Auth Token:  
ngrok authtoken YOUR_NGROK_KEY

Start the Streamlit app:  
!nohup streamlit run crowd.py --server.port 8501 --server.headless true &

Expose it with Ngrok:  
from pyngrok import ngrok  
public_url = ngrok.connect(8501)  
print("Your app is live at:", public_url)

---

## How It Works

1. Upload a video in the dashboard.  
2. Model generates density maps and counts.  
3. Alert is triggered when the threshold is crossed.  
4. Email is sent with the crowd count.

Note: Training for multiple epochs increases the accuracy of the output.

---

## Author

**Bhuvan M J**  
AI & ML Engineer | Passionate about Computer Vision & Deep Learning  
GitHub: [https://github.com/Bhuvanmj](https://github.com/Bhuvanmj)
