Edge-Optimized Weed Detection System

## Overview

This project implements an edge-optimized computer vision pipeline for real-time weed detection as part of the perception subsystem of a university autonomous rover. The objective is to accurately detect weeds under varying field conditions while maintaining low computational overhead for deployment on resource-constrained edge devices.

The complete machine learning workflow—from dataset preparation to deployment-ready inference—was developed to enable reliable agricultural object detection in real-world environments.

---

## Features

- End-to-end object detection pipeline
- Dataset preprocessing and annotation conversion
- Data augmentation for improved model generalization
- Model training and validation using Ultralytics YOLO
- Evaluation using Precision, Recall, and mAP metrics
- Optimized inference for edge deployment
- Integration with the rover perception pipeline

---

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## Project Workflow

```
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
YOLO Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Edge Optimization
        │
        ▼
Real-Time Inference
        │
        ▼
Autonomous Rover Integration
```

---

## Repository Structure

```
Weed-Detection/
│
├── datasets/
├── models/
├── src/
├── results/
├── docs/
└── README.md
```

---

## Current Status

✔ Dataset preparation completed

✔ Model training pipeline completed

✔ Evaluation pipeline completed

✔ Edge deployment optimization completed

✔ Integrated as the computer vision subsystem of a university autonomous rover

---

## Future Improvements

- Expand the dataset with additional crop and weed species
- Improve robustness under varying illumination and weather conditions
- Evaluate lightweight detection architectures for faster inference
- Deploy and validate the system on embedded edge hardware

---
