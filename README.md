🪖 Helmet Detection and Face Capture System
📌 Overview

This project is a real-time computer vision system that detects whether a person is wearing a helmet or not. If a no-helmet violation is detected, the system automatically captures and stores the person's face image for monitoring and safety compliance.

It is built using YOLO object detection and OpenCV-based face extraction pipeline.

🚀 Features
Real-time helmet vs no-helmet detection
Automatic face detection and cropping on violation
Stores captured face images locally
Works on video streams / webcam input
Lightweight and fast inference pipeline
🧠 Tech Stack
Python
YOLO (Object Detection Model)
OpenCV (Image Processing)
Roboflow Dataset
Google Colab (Model Training)
Caffe Model (Face Detection DNN)
📂 Project Workflow
Input Video / Webcam
        ↓
YOLO Object Detection
        ↓
Check Class = No Helmet
        ↓
Face Detection (OpenCV DNN / Caffe)
        ↓
Crop Face Region
        ↓
Save Image to Dataset Folder
📊 Model Performance
Precision: ~0.79
Recall: ~0.83
mAP@50: ~0.85
mAP@50-95: ~0.70
🗂 Dataset
Dataset prepared using Roboflow
Classes:
Helmet
No Helmet
Driver
Bicyclist
