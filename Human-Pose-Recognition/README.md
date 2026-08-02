# Human Pose Recognition System

## Overview

This project implements a machine learning pipeline for human pose recognition using MediaPipe Pose Landmarker and scikit-learn. It extracts human skeletal landmarks from images, engineers geometric angle features, and classifies yoga poses using a Multi-Layer Perceptron classifier.

The current trained model recognizes five poses:

- Downdog
- Goddess
- Plank
- Tree
- Warrior2

## Features

- Human pose landmark extraction using MediaPipe Pose Landmarker
- Automatic dataset generation from pose image folders
- Geometric feature engineering from skeletal landmarks
- Pose classification using an MLP classifier
- Image-based inference
- Live webcam-based pose recognition and visual feedback

## Technologies Used

- Python
- MediaPipe Pose Landmarker
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Joblib

## Setup

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

## Train

The dataset should be organized as one folder per pose:

```text
YogaPoses/
  Downdog/
  Goddess/
  Plank/
  Tree/
  Warrior2/
```

Build the feature CSV:

```powershell
$env:PYTHONPATH = ".\src"
python -c "from dataset_builder import DatasetBuilder; DatasetBuilder().build(r'C:\Users\Dhruv\Desktop\YogaPoses', 'data/features.csv')"
```

Train the classifier:

```powershell
python src\train.py
```

## Run Live

Start webcam-based pose recognition:

```powershell
python src\live.py
```

Use a different camera index if needed:

```powershell
python src\live.py --camera 1
```

Press `q` to quit the video window.

## Run On A Test Image

Place a test image at `test.jpg`, then run:

```powershell
python src\main.py
```

## Project Workflow

```text
Input Images or Webcam Frames
      |
      v
MediaPipe Pose Landmarker
      |
      v
Pose Skeleton
      |
      v
Angle Feature Extraction
      |
      v
MLP Classifier
      |
      v
Pose Prediction and Feedback Overlay
```
