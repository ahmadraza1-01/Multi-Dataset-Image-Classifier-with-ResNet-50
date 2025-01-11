# Multi-Dataset-Image-Classifier-with-ResNet-50
This repository contains the implementation of a **multi-dataset image classification pipeline** designed to handle domain shifts and improve accuracy across diverse datasets. The project leverages **ResNet-50** for feature extraction, pseudo-labeling for unlabeled datasets, and label smoothing for robust classification.  

## Features  
- **Feature Extraction**: Utilizes ResNet-50, a pre-trained convolutional neural network, for extracting 2048-dimensional feature vectors from input images.  
- **Custom Classification Model**: Includes a feedforward neural network with ReLU activation for multi-class classification.  
- **Pseudo-Labeling**: Enables training on unlabeled datasets by generating pseudo-labels using model predictions.  
- **Label Smoothing**: Reduces overconfidence in predictions and enhances generalization.  
- **Fine-Tuning**: Incrementally improves accuracy by training on multiple datasets while mitigating overfitting.  

## Methodology  
1. **Feature Extraction**:  
   - Preprocess input images by normalizing them with ImageNet mean and standard deviation.  
   - Extract features using the truncated ResNet-50 model, retaining only its feature extraction layers.  

2. **Classification Pipeline**:  
   - Input extracted features to a feedforward neural network with a hidden layer of 256 neurons (ReLU activation).  
   - Output layer predicts across 10 classes using softmax activation.  

3. **Training Strategy**:  
   - Supervised training on labeled datasets (Dataset D1).  
   - Pseudo-labeling for unlabeled datasets (D2 to D10).  
   - Fine-tune the model for 5 epochs to prevent overfitting.  

## Results  
- **Accuracy Improvements**: Achieved progressive accuracy enhancements across 10 datasets.  
- **Generalization Strength**: Label smoothing and pseudo-labeling contributed to better handling of noisy labels and domain shifts.  
- **Limitations**: Performance degradation observed for certain datasets due to domain differences and pseudo-label quality.  

## Strengths  
- Effective use of transfer learning with ResNet-50.  
- Robust classification through pseudo-labeling and label smoothing.  
- Scalability to multiple datasets with fine-tuning techniques.  

## Limitations  
- Dependency on pseudo-label quality impacting accuracy.  
- Reduced performance on datasets with significant domain shifts.  

## Conclusion  
The methodology demonstrates the power of transfer learning, pseudo-labeling, and fine-tuning to address image classification challenges across diverse datasets.  

---  
