# 🟢 Crowd Counting Dashboard with Smart Alerts (CSRNet)

This project provides a complete crowd counting solution using the **CSRNet** model for real-time density estimation in video streams. The system is deployed as a dynamic Streamlit web application and features **smart alerting** via email (SMTP) and SMS (Twilio) when a specified crowd threshold is exceeded.

***

## 💾 Core Components

The application is built around the following files and technologies:

* **CrowdTask.ipynb (The Jupyter Notebook):** Contains the setup, data loading, model definition (CSRNet), training/loading logic, and the execution environment setup (including the installation of necessary libraries and Ngrok configuration).
* **/content/streamlit_app.py:** The main Python script defining the Streamlit user interface, the video processing loop, the prediction logic, and the alerting functions.
* **CSRNet Model:** A pre-trained PyTorch model based on VGG16 with a dilated convolutional backend, used to predict crowd density maps.
* **Alerting Services:** Utilizes **SMTP (via standard Python libraries and environment variables)** for email and the **Twilio API** for SMS notifications.

## 🚀 Setup and Requirements

### 1. Model Checkpoint

The system requires the pre-trained CSRNet model checkpoint to be accessible.

* **Required Path:** /content/drive/MyDrive/ShanghaiTech/best_csrnet.pth

### 2. Python Environment

All necessary libraries are typically installed in the initial notebook cells. Ensure you have the following packages:

* PyTorch (torch, torchvision, torchaudio)
* Streamlit
* OpenCV (opencv-python-headless)
* Matplotlib, NumPy, Pillow, tqdm, SciPy
* Ngrok (pyngrok)
* Twilio
* yagmail

### 3. Ngrok Authentication

Since the application is run from a remote environment (like Google Colab), Ngrok is used to create a public URL.

* Set your **Ngrok Auth Token** using a notebook cell command before running the app.

## 🔔 Smart Alert Configuration

The system uses **environment variables** (typically set via notebook cells or Colab's *Secrets* feature) for authentication.

### A. Email Alerts (SMTP/Gmail)

The system is configured to send emails using Gmail's SMTP service (port 587, using TLS).

* **Required Environment Variables:**
    * SENDER_EMAIL: Your Gmail address.
    * SENDER_PASSWORD: An **App Password** generated from your Google Account if 2FA is enabled (recommended).

### B. SMS Alerts (Twilio)

The system uses the Twilio API for sending SMS alerts.

* **Required Environment Variables:**
    * TWILIO_ACCOUNT_SID: Your Twilio Account SID.
    * TWILIO_AUTH_TOKEN: Your Twilio Auth Token.
    * TWILIO_PHONE_NUMBER: Your Twilio sender phone number (must be in E.164 format, e.g., +15551234567).

***

## 💻 Running the Application

1.  **Preparation:** Execute all setup and model loading cells in the Jupyter Notebook to install dependencies, load the model, and set the environment variables.
2.  **Launch:** The Streamlit application is launched in the background via a notebook cell command:
    * *python -m streamlit run /content/streamlit\_app.py --server.port 8501 --server.headless true &*
3.  **Access:** A public URL is generated using Ngrok, which is displayed in the notebook output. Access this URL to use the dashboard.

## 🌐 Dashboard Features

The Streamlit dashboard offers a user-friendly interface for continuous crowd monitoring:

* **Video Upload:** Users can upload an MP4 video file for analysis.
* **Crowd Threshold (Slider):** Defines the critical count level that triggers alerts.
* **Frame Interval (Input):** Allows skipping frames to speed up processing (e.g., process every 5th frame).
* **Alert Configuration:** Checkboxes to enable Email/SMS and text inputs for recipient addresses/numbers (E.164 format for phones).
* **Test Alerts:** Buttons to send sample Email or SMS alerts to validate credentials and recipients.
* **Real-time Visualization:** Displays the current video frame alongside the predicted **Density Map** (heatmap visualization).
* **Metrics:** Shows the current crowd count, along with peak and average counts over the video stream.
* **Alert Logic:** Displays a warning when the count exceeds the threshold. Alerts (email/SMS) are sent **only once** per event (i.e., only when the count *crosses* the threshold from below).
