# Leaf Classification using Deep Neural Networks

## Overview

This project focuses on classifying leaf images to predict plant species using Deep Neural Networks. We experimented with various techniques, including custom networks and transfer learning with fine-tuning, to achieve high accuracy in leaf classification.

## Dataset

- 17,728 images of leaves on a black background
- 14 plant species (apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, tomato)
- 256x256 resolution JPEG images
- Unbalanced class distribution

## Key Features

- Data analysis and preparation
- Data augmentation techniques
- Transfer learning with pre-trained models
- Custom CNN architectures
- Performance comparison of various models

## Methods

### Data Preparation
- Dataset splitting (training, validation, test)
- Oversampling and class weighting to address class imbalance
- Data augmentation (rotation, zoom, flipping, translations, shears)

### Models
- Transfer learning: VGG16, ResNet152v2, EfficientNetV7, DenseNet-201, Xception, ResNet50v2
- Custom deep convolutional neural networks
- Experiments with autoencoder-based feature extraction

### Training
- Cross-validation via hold-out
- Two-step training for transfer learning models
- Learning rate adjustment

## Results

Best performing models on remote test data:

| Model         | Accuracy |
|---------------|----------|
| DenseNet-201  | 0.9075   |
| EfficientNetV7| 0.9038   |
| ResNet152v2   | 0.8736   |
| ResNet50V2    | 0.8264   |
| Xception      | 0.7264   |

## Conclusions

Transfer learning models with fine-tuning achieved the best performance in terms of accuracy and training time, given the limited training resources. DenseNet-201 showed the highest accuracy of 90.75% on the test set.

## Tools Used

- TensorFlow
- Keras
- Scikit-Learn
- Jupyter Notebook

## Authors

- Marco Domenico Buttiglione
- Luca De Martini
- Giulia Forasassi

Politecnico di Milano

## Date

November 30, 2021

---

For more detailed information about the methodology, experiments, and findings, please refer to the full project report.
