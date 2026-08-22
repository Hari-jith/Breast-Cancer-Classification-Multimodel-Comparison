# Breast Cancer Classification: Multimodel Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)

## 📋 Project Overview
This project focuses on the binary classification of breast cancer using **Ultrasound Breast Images**. The goal is to distinguish between **Benign** and **Malignant** tumors. 

To achieve this, we implemented and compared four state-of-the-art Convolutional Neural Network (CNN) architectures utilizing **Transfer Learning**. The models were evaluated based on Accuracy, Precision, Recall, F1-Score, and AUC-ROC, with the ultimate goal of identifying the best performing model for medical image diagnostics.

## 🚀 Models Evaluated
We evaluated the following pre-trained models (using ImageNet weights) with custom classification heads (Global Average Pooling + Dense + Dropout):

1. **VGG19**
2. **DenseNet121** (*Selected as Best Model*)
3. **EfficientNetB0**
4. **InceptionV3**

## 📊 Dataset
- **Source:** [Ultrasound Breast Images for Breast Cancer](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Classes:** Benign, Malignant
- **Image Size:** 224x224 pixels
- **Train Set:** 8,116 images
- **Validation Set:** 900 images
- **Preprocessing:** Data Augmentation (Rotation, Zoom, Shift, Flip) applied to the training set to improve generalization.

## 🛠️ Methodology
1. **Data Preprocessing:** Images were resized and normalized. Augmentation techniques were used to prevent overfitting.
2. **Transfer Learning:** The convolutional base of each model was frozen to leverage pre-learned features. 
3. **Custom Classifier:** A `GlobalAveragePooling2D` layer followed by a `Dense` layer (256 units, ReLU), `Dropout` (0.4), and a final `Sigmoid` output layer was added on top.
4. **Training:** Each model was trained for **10 epochs** using the Adam optimizer (learning rate \(1e^{-4}\)) and Binary Crossentropy loss.

## 📈 Results & Performance Comparison

The models were evaluated on a separate validation set of 900 images. Below is the summary of their performance:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG19** | 0.7922 | 0.8142 | 0.6900 | 0.7470 | 0.8769 |
| **DenseNet121** | **0.8689** | 0.8543 | **0.8500** | **0.8521** | 0.9439 |
| **EfficientNetB0** | 0.5556 | 0.0000 | 0.0000 | 0.0000 | 0.6647 |
| **InceptionV3** | 0.8644 | **0.9633** | 0.7225 | 0.8257 | **0.9706** |

*Note: EfficientNetB0 failed to converge, likely due to a learning rate mismatch with the pre-trained weights, resulting in a biased model that predicts only the majority class.*

### 📂 Result Visualizations
All generated plots (Training Curves, Confusion Matrices, ROC Curves, and Model Comparison) are located in the `results/` folder:
- `results/VGG19_*`
- `results/DenseNet121_*`
- `results/EfficientNetB0_*`
- `results/InceptionV3_*`

![Model Comparison](results/performance_comparison.png)

## 🏆 Final Selection: DenseNet121
While InceptionV3 achieved the highest AUC and Precision, **DenseNet121** was selected as the final model due to its superior **F1-Score (0.8521)** and **Recall (0.8500)**. 
In medical diagnostics, minimizing false negatives (missed Malignant cases) is critical. DenseNet121 provides the best balance between identifying malignant cases and maintaining overall accuracy.

Below is a sample of random predictions using the final DenseNet121 model:

![Random Predictions](results/densenet_random_predictions.png)

## 🛡️ Requirements
To run this project, install the following dependencies:
```bash
pip install requirements.txt
```

## 🧑‍💻 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Hari-jith/Breast-Cancer-Classification-Multimodel-Comparison.git
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
3. Update the `train_dir` and `val_dir` paths in the notebook.
4. Run all cells in `Breast_Cancer_Classification.ipynb`.

## 📜 License
This project is licensed under the MIT License. 

## 👤 Author

Harijith M M 
- GitHub: [Hari-jith](https://github.com/Hari-jith)
