# Diabetic Retinopathy Detection using Transfer Learning

## Overview
This project focuses on detecting diabetic retinopathy from retinal fundus images using deep learning. The goal is to automate classification of disease severity using transfer learning on pretrained convolutional neural networks.

---

## Problem Statement
Diabetic Retinopathy is a leading cause of blindness among diabetic patients. Early detection is crucial but requires expert analysis. This project aims to build a deep learning-based system to classify retinal images into different severity levels.

---

## Approach

### Data Preprocessing
- Resized all images to **224 × 224**
- Normalized pixel values (rescale = 1/255)
- Used `ImageDataGenerator` for efficient data loading
- Applied **80–20 train-validation split**

### Model Development
Implemented transfer learning using:
- MobileNetV2  
- ResNet50  
- EfficientNetB0  

For each model:
- Loaded ImageNet pretrained weights  
- Froze base layers (feature extractor)  
- Added custom classification head:
  - GlobalAveragePooling  
  - Dropout (0.3)  
  - Dense layer with Softmax activation  

### Training Configuration
- Optimizer: Adam (learning rate = 1e-4)  
- Loss Function: Categorical Crossentropy  
- Metric: Accuracy  
- Callbacks:
  - EarlyStopping  
  - ReduceLROnPlateau  
  - ModelCheckpoint  

---

## Results

| Model           | Training Accuracy | Validation Accuracy |
|----------------|------------------|--------------------|
| MobileNetV2    | 70.9%            | 73.8%              |
| ResNet50       | 52.0%            | 49.2%              |
| EfficientNetB0 | 48.8%            | 49.2%              |

**MobileNetV2 achieved the best performance.**

---

## Evaluation Insights
- Accuracy alone is not sufficient due to class imbalance  
- Classification report shows poor performance for minority classes  
- Confusion matrix indicates bias toward dominant class (No_DR)  
- Model struggles to generalize across all severity levels  

---

## Key Learnings
- Transfer learning improves performance on small datasets  
- Simpler models can outperform deeper models in limited data scenarios  
- Class imbalance significantly impacts model evaluation  
- Multiple evaluation metrics are required beyond accuracy  

---

## Limitations
- Class imbalance not handled  
- Pretrained layers fully frozen (no fine-tuning)  
- Limited training epochs  

---

## Future Improvements
- Apply class balancing techniques (class weights / oversampling)  
- Fine-tune pretrained layers  
- Increase dataset size  
- Add explainability (Grad-CAM)  

---

## Dataset
Dataset sourced from Kaggle (Diabetic Retinopathy Detection).  
Not included in the repository due to size constraints.

---

## How to Run

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook retinopathy_notebook_01.ipynb
```

---

## Project Structure

```
diabetic-retinopathy-detection/
│
├── retinopathy_notebook_01.ipynb
├── README.md
├── requirements.txt
├── results/
└── data/ (not included)
```

---

## Conclusion
This project demonstrates the use of transfer learning for medical image classification. Among the tested models, MobileNetV2 performed best, showing that lightweight architectures can generalize better on smaller datasets.
