# Human Pose Recognition System

## Overview

This project implements a machine learning pipeline for human pose recognition using MediaPipe Pose Landmarker and scikit-learn. The system extracts human skeletal landmarks from images, engineers geometric features such as joint angles and limb orientations, and classifies poses using a Multi-Layer Perceptron (MLP) classifier.

The project is currently being extended into a multimodal human activity recognition system by integrating smartwatch sensor data (IMU) with vision-based pose features to improve recognition accuracy and robustness.

---

## Features

- Human pose landmark extraction using MediaPipe Pose Landmarker
- Automatic dataset generation from pose images
- Geometric feature engineering from skeletal landmarks
- Pose classification using an MLP classifier
- Automated preprocessing, training, and inference pipeline
- Ongoing multimodal integration with smartwatch IMU data

---

## Technologies Used

- Python
- MediaPipe Pose Landmarker
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## Project Workflow

```
Input Images
      │
      ▼
MediaPipe Pose Landmarker
      │
      ▼
33 Pose Landmarks
      │
      ▼
Feature Engineering
      │
      ▼
Dataset Generation
      │
      ▼
MLP Classifier Training
      │
      ▼
Pose Prediction
```

### Current Extension

```
Camera Images               Smartwatch IMU
       │                           │
       ▼                           ▼
 Pose Landmarks            Sensor Processing
       │                           │
       └──────────────┬────────────┘
                      ▼
             Feature Fusion
                      ▼
      Human Activity Recognition
```

---

## Current Status

✔ Pose landmark extraction completed

✔ Feature engineering pipeline completed

✔ Dataset generation completed

✔ MLP classifier training completed

✔ Automated inference pipeline completed

🔄 Multimodal smartwatch integration in progress

---

## Future Improvements

- Real-time human activity recognition
- Advanced multimodal feature fusion techniques
- Deep learning-based sequence models
- Edge deployment on embedded hardware
- Support for additional activities and larger datasets

---
