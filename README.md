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

## Dataset
Ultrasound Breast Cancer Dataset (Kaggle)

Dataset Link: https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer/data

### Sample Images of Dataset

<img width="774" height="405" alt="sample data" src="https://github.com/user-attachments/assets/bb222dbd-f19d-4974-88d7-452a5ec30eaf" />

## Data Preprocessing
The following preprocessing steps were applied:

  -Image resizing to 224 × 224
  
  -Pixel normalization (rescale 1/255)
  
  -Data augmentation:
  
    -Random rotation
    -Zoom
    -Horizontal flip
    
  -Binary label encoding

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

## Results

| Model          | Accuracy | Precision | Recall   | F1       | AUC      |
| -------------- | -------- | --------- | -------- | -------- | -------- |
| VGG19          | 0.79     | 0.81      | 0.69     | 0.75     | 0.88     |
| DenseNet121    | **0.87** | 0.85      | **0.85** | **0.85** | 0.94     |
| EfficientNetB0 | 0.56     | 0.00      | 0.00     | 0.00     | 0.66     |
| InceptionV3    | 0.86     | **0.96**  | 0.72     | 0.83     | **0.97** |


## Best Model
DenseNet121 achieved the best overall performance:
- Accuracy: 86.8%
- F1-score: 0.85
- AUC: 0.94
<img width="863" height="393" alt="output" src="https://github.com/user-attachments/assets/40e77c08-c2e5-4315-9202-3292c3df9fdd" />


## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
