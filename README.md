# Breast-Cancer-Classification-Multimodel-Comparison
Deep learning based breast cancer classification using ultrasound images


## Project Overview

Breast cancer is one of the most prevalent cancers among women worldwide. Early and accurate diagnosis significantly improves treatment outcomes. This project presents a deep learning-based approach for classifying breast ultrasound images into benign and malignant categories using multiple pre-trained Convolutional Neural Network (CNN) architectures.

A comparative analysis is performed on four state-of-the-art models to identify the most reliable architecture for this task.

## Models Used
- VGG19
- DenseNet121
- EfficientNetB0
- InceptionV3

## Best Model
DenseNet121 achieved the best overall performance:
- Accuracy: 86.8%
- F1-score: 0.85
- AUC: 0.94

## Dataset
Ultrasound Breast Cancer Dataset (Kaggle)

Dataset Link: https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer/data

### Sample Images of Dataset

<img width="774" height="405" alt="sample data" src="https://github.com/user-attachments/assets/bb222dbd-f19d-4974-88d7-452a5ec30eaf" />

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Technologies
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
