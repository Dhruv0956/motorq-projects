# Exoplanet Transit Detection

## Overview

This project focuses on the detection and analysis of exoplanet transit signals using real photometric observations from **NASA's Transiting Exoplanet Survey Satellite (TESS)**. It aims to build an end-to-end scientific data analysis pipeline capable of processing astronomical light curves, identifying periodic transit events, and validating potential exoplanet candidates through statistical techniques. NASA's TESS mission continuously observes stellar brightness to detect the characteristic dips produced when a planet passes in front of its host star. :contentReference[oaicite:0]{index=0}

The current implementation analyzes observations of **TIC 441738827**, with the project being actively expanded into a generalized exoplanet detection pipeline.

---

## Features

- Processing of real NASA TESS light curve data
- Time-series visualization and exploratory analysis
- Light curve preprocessing and normalization
- Transit event identification
- Statistical analysis of periodic signals
- Interactive Jupyter Notebook workflow
- Ongoing development of a reusable detection pipeline

---

## Technologies Used

- Python
- Astropy
- Lightkurve
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

## Current Workflow

```
NASA TESS Light Curve
          │
          ▼
Data Acquisition
          │
          ▼
Light Curve Preprocessing
          │
          ▼
Time-Series Analysis
          │
          ▼
Transit Signal Detection
          │
          ▼
Candidate Validation
          │
          ▼
Visualization & Interpretation
```

---

## Repository Structure

```
Exoplanets/
│
├── TIC 441738827.ipynb
├── README.md
└── assets/          (future)
```

---

## Current Status

✔ Real NASA TESS data analysis

✔ Time-series preprocessing

✔ Transit visualization

🔄 Generalized detection pipeline in development

🔄 Statistical transit detection and candidate validation being implemented

---

## Future Work

- Implement the Box Least Squares (BLS) algorithm for automated transit detection
- Analyze multiple TESS targets through a unified pipeline
- Improve preprocessing and detrending of light curves
- Integrate statistical validation techniques for candidate selection
- Explore machine learning approaches for transit candidate classification

---

## Data Source

This project uses publicly available photometric observations from **NASA's Transiting Exoplanet Survey Satellite (TESS)** mission. TESS provides high-precision stellar light curves for the discovery and characterization of exoplanets through the transit method. :contentReference[oaicite:1]{index=1}

---
