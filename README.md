MUlti-MOdel Heart abnormality Ai model using PCG signals and MFCC advanced Features Extraction
🫀 PCG Heart Sound Classification using MFCC and Multi-Model Learning

This repository presents a research-oriented system for automatic abnormal heart sound detection using phonocardiogram (PCG) signals.
The project implements a comparative framework using classical machine learning, deep learning, and temporal deep learning models based on MFCC features.

📌 Project Motivation

Phonocardiograms are low-cost, non-invasive signals used for cardiac screening, but they are:

Noisy and non-stationary

Difficult to interpret manually

Variable in length

This project explores whether Mel-Frequency Cepstral Coefficients (MFCCs) can provide a robust and unified feature representation across multiple learning paradigms.

🧠 Models Implemented

The same MFCC representation is evaluated using three fundamentally different models:

Model	Learning Paradigm	Purpose
Gradient Boosting	Classical ML	Statistical MFCC patterns
CNN	Deep Learning	Spatial MFCC structures
CNN-LSTM	Temporal DL	Sequential heart sound dynamics

This comparative design makes the project research-driven rather than application-only.

📂 Dataset

PhysioNet / CinC Challenge 2016

PCG recordings labeled as Normal or Abnormal

Publicly available, widely used in cardiac research

⚙️ Methodology
1. Signal Preprocessing

Resampling to 2 kHz

Segmentation into fixed-length windows

Noise-robust normalization

2. Feature Extraction

MFCC extraction (40 coefficients)

Time–frequency representation

Statistical aggregation for ML models

3. Model Training

Segment-level learning (Gradient Boosting, CNN)

Record-level temporal learning (CNN-LSTM)

Binary classification: Normal vs Abnormal

4. Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

ROC-AUC (primary metric due to class imbalance)
