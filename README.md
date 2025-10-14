# DeepVision Crowd Monitoring
  # Project Overview

The DeepVision Crowd Monitoring project is a computer vision system designed to estimate the number of people in images or videos and visualize crowd density using deep learning. It can also generate a density map to show areas with more people.
This project was built using PyTorch for model training, OpenCV for image/video processing, and Streamlit for a simple interactive dashboard.
deepvision-crowd/
├── notebooks/      ← Colab notebook(s) containing all the code, training, and testing
├── app/            ← Streamlit app (app.py) to interactively estimate crowd count
└── demo/           ← Screenshots or demo outputs showing results
3 Files Description
# notebooks/
Contains the Colab notebook of code where:
The dataset is loaded and preprocessed
Crowd counting model (CNN) is defined and trained
Sample images are tested and density maps are visualized
# app/
app.py is a Streamlit dashboard where you can:
Upload an image
Estimate the crowd count
Visualize the density map overlay
Receive alerts if crowd exceeds a threshold
# demo/
Contains screenshots or exported demo outputs showing:
Sample images
Density maps
Crowd count estimation
These images demonstrate how the project works without needing to run the model.
# How to Use
Run the notebook in Google Colab:
 Install required packages: torch, torchvision, opencv-python, matplotlib, etc.
 Execute the cells to preprocess data and train the model.
Use the Streamlit app:
 Open a terminal or Colab cell to run:
streamlit run app.py
 Upload an image using the sidebar
 Click “Estimate Crowd” to see the crowd count and density map
View demo outputs:
Open the images in the demo/ folder to see sample results.
# Key Features
 Estimates total people in an image
 Generates crowd density maps (heatmaps)
 Streamlit app provides interactive dashboard
 sends overcrowding alerts (if configured with email)
# Technologies Used
 Python – main programming language
 PyTorch – for building and training model
 OpenCV – image and video processing
 Matplotlib – for visualizations
 Streamlit – interactive dashboard and GUI
