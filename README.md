# DeepVision Crowd Monitor: AI for Density Estimation and Overcrowding Detection

🎯 Project Overview

The DeepVision Crowd Monitor is an AI-powered computer vision system designed to estimate crowd density and detect overcrowding situations in real time.
Using deep learning—specifically Convolutional Neural Networks (CNNs)—the model can analyze images or live video frames, count the number of people, and visualize density maps showing which areas are more crowded.

This kind of system is highly useful for public safety, event management, transportation hubs, and smart city monitoring, where controlling crowd flow is critical to prevent hazards.

## 📘 Overview

This project implements an AI-driven system for real-time crowd density estimation and overcrowding detection using deep learning. 
It leverages the ShanghaiTech crowd dataset and convolutional neural network (CNN) models trained in PyTorch to predict 
density maps and estimate crowd counts from surveillance images.

The notebook is divided into multiple tasks:
- **Task 1:** Dataset loading, preprocessing, and ground-truth density map generation.
- **Task 2:** Custom PyTorch dataset class creation and image transformation setup.
- **Task 3:** Model training, validation, and performance visualization.
- **Task 4:** Deployment and alert mechanisms (using Streamlit/Twilio for live monitoring).

---

🔍 Core Idea

Traditional crowd counting methods struggle in dense scenes where individuals overlap or are partially visible.
To overcome this, DeepVision Crowd Monitor uses a density map estimation approach, where the network learns to output a pixel-wise density map instead of discrete counts.

Each pixel value in the density map represents the likelihood of a person being present there.
By integrating (summing) all pixel values, the system estimates the total crowd count accurately—even in highly congested environments.

---

## ⚙️ Setup Instructions

### 1. Requirements
Ensure Python ≥ 3.8 is installed, then install dependencies using:
pip install torch torchvision torchaudio opencv-python-headless numpy scipy scikit-learn matplotlib pillow streamlit twilio pyngrok tqdm

### 2. Dataset
- Download the **ShanghaiTech Crowd Counting Dataset (Part A)** from its official source or Kaggle.
- Place the dataset in a Google Drive directory (if running on Google Colab) or a local path structured as:
/path_to_dataset/
 ├── train_data/
 │    ├── images/
 │    └── ground_truth/
 └── test_data/
      ├── images/
      └── ground_truth/

### 3. Run the Notebook
Open and execute the notebook:
- On Google Colab → Upload the notebook and connect your Google Drive.
- On local Jupyter → Run cells sequentially after adjusting dataset paths.

---

## 🧩 Model & Training

- **Architecture:** Convolutional Neural Network with Gaussian density map estimation.
- **Loss Function:** Mean Squared Error (MSE) between predicted and ground-truth density maps.
- **Optimizer:** Adam (default learning rate = 1e-5)
- **Epochs:** 50 (configurable)
- **Batch Size:** 8

---

## 📊 Results & Outputs

The model outputs:
- Density maps for input crowd images.
- Predicted crowd count per image.
- Visual plots comparing predicted vs actual densities.
- Streamlit-based demo for live crowd monitoring (optional).
- Twilio alert integration for overcrowding notifications.

---

## 🚀 Deployment (Optional)

To run the Streamlit app:
streamlit run app.py

If ngrok or Twilio is configured, the app can send SMS alerts when crowd count exceeds a threshold.

---

## 📁 File Structure

DeepVision_Crowd_Monitor.ipynb     # Main Jupyter Notebook
app.py                             # Optional Streamlit app (if included)
/data                              # Dataset folder (ShanghaiTech)
/models                            # Saved trained models
/results                           # Output visualizations and predictions

---

## 👨‍💻 Author
Developed by **[S.Vishnu Varshan Reddy]**
Third-year Engineering Student — Machine Learning and AI specialization

---

## 📄 License
This project is open for educational and research use. Please cite appropriately if reused.
