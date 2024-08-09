# Machine-Learning-2
<div align="center">
  <a href="https://unige.it/en/">
    <img src="./logounige.jpg" width="20%" height="20%" title="University of Genoa" alt="University of Genoa">
  </a>
</div>

<h1 align="center"> Facial Emotion Recognition for Enhancing Human-Robot Interaction </h1>

> **Authors:**
> - *Roumaissa Benkredda*  
> - *Seyed Amir Mohamad Hosseini Rad*
>
> **Course:** Machine Learning and Data Analysis  
> **Department:** DIBRIS - University of Genova  
> **Professor:** Oneto Luca
> **Submission Date:** June 24,2024

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Dataset and Preprocessing](#dataset-and-preprocessing)
4. [Model Architecture](#model-architecture)
    * [Initial Model](#initial-model)
    * [Model 2](#model-2)
    * [Model 3](#model-3)
5. [Training Process](#training-process)
6. [Results and Evaluation](#results-and-evaluation)
7. [Smart Solution for Dataset Filtering](#smart-solution-for-dataset-filtering)
8. [Installation Instructions](#installation-instructions)
9. [References](#references)
10. [Appendices](#appendices)

---

<a name="introduction"></a>

## Introduction

This repository contains the code and resources for a project focused on enhancing human-robot interaction through facial emotion recognition. The project uses Convolutional Neural Networks (CNNs) built with Keras and TensorFlow to classify facial emotions from images. The primary objective is to develop a robust system capable of accurately recognizing facial expressions, which can be integrated into robotic systems for better interaction with humans.

---

<a name="project-overview"></a>

## Project Overview

### Objective:
The main goal of this project is to create a facial emotion recognition system to improve human-robot interaction. The system leverages deep learning techniques to classify emotions, making robots more responsive and empathetic towards human emotions.

### Importance:
Emotion recognition is crucial in the development of interactive systems that can adapt to human emotions, making interactions more natural and effective.

### Tools and Technologies:
- **Programming Language:** Python
- **Deep Learning Framework:** Keras with TensorFlow backend
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

<a name="dataset-and-preprocessing"></a>

## Dataset and Preprocessing

### Dataset:
The dataset used in this project was sourced from Kaggle, containing 18,663 grayscale images categorized into seven emotion classes: anger, disgust, fear, happiness, neutral, sadness, and surprise. Each image is standardized to a size of 48x48 pixels.

### Preprocessing Steps:
1. **Grayscale Conversion:** Ensures consistency and reduces computational complexity.
2. **Image Augmentation:** Using Keras's `ImageDataGenerator` to apply transformations like rotation, scaling, and flipping to enhance model robustness.

### Directory Structure:
The data is organized into training and validation sets, with subdirectories for each emotion class. This structure allows Keras to automatically label the images during training.

---

<a name="model-architecture"></a>

## Model Architecture

### Initial Model
- **Batch Size:** 128
- **CNN Layers:** 3
- **Dense Layers:** 2
- **Activation Function:** Sigmoid
- **Pooling:** MaxPooling2D
- **Optimizer:** SGD

### Model 2
- **Batch Size:** 256
- **CNN Layers:** 5
- **Dense Layers:** 3
- **Activation Function:** ReLU
- **Pooling:** MaxPooling2D
- **Optimizer:** Adam with learning rate 0.001
- **Batch Normalization:** Added

### Model 3
- **Batch Size:** 128
- **CNN Layers:** 4
- **Dense Layers:** 3
- **Activation Function:** ReLU
- **Pooling:** AveragePooling2D
- **Optimizer:** Adam with learning rate 0.0001
- **Dropout Layers:** Added to prevent overfitting

---

<a name="training-process"></a>

## Training Process

### Training Execution:
- **Batch Size:** Controls the number of samples processed before updating the model's weights.
- **Learning Rate:** Adjusted to optimize training speed and model accuracy.
- **Callbacks Used:**
  - `ModelCheckpoint`: Saves the best model based on validation accuracy.
  - `EarlyStopping`: Stops training early if the model's performance stops improving.
  - `ReduceLROnPlateau`: Reduces the learning rate when validation loss stagnates.

### Training Metrics:
- **Accuracy:** Measures the percentage of correct predictions.
- **Loss:** Measures the error between predicted and true labels.

---

<a name="results-and-evaluation"></a>

## Results and Evaluation

### Final Model Performance:
- **Training Accuracy:** 72.82%
- **Validation Accuracy:** 70%
- **Training Loss:** 0.7481
- **Validation Loss:** 0.8792

### Evaluation:
- **Model 3:** Demonstrated significant improvements by addressing overfitting, leading to a more stable validation accuracy and better generalization.

### Confusion Matrix:
A confusion matrix was generated to visualize the classification performance across all emotion classes, helping identify areas of misclassification.

---

<a name="smart-solution-for-dataset-filtering"></a>

## Smart Solution for Dataset Filtering

### Problem:
The dataset contained mislabeled images where the emotion was not detectable, leading to improper training and model errors.

### Solution:
Developed a code utilizing the DeepFace library to automatically filter out incorrect images based on pre-trained models. This significantly improved the dataset quality, resulting in better model performance.

---

<a name="installation-instructions"></a>

## Installation Instructions

### Prerequisites:
- Python 3.x
- TensorFlow and Keras
- Additional libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

### Setup:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-recognition.git](https://github.com/Roumaissa-BENKREDDA/Machine-Learning-2.git
   cd emotion-recognition

### Prepare the Dataset:
Place your dataset in the Dataset/ directory as described in the project structure.

### Run the Model:
Train the model using the provided Jupyter notebook or Python scripts.

### View the Results:
After training, you can view the results, such as accuracy and loss, by using the plots generated in the notebook or the Python script. Additionally, you can analyze the confusion matrix to understand the model's performance across different emotion classes.

### Use the Pre-trained Model:
If you wish to use the pre-trained model, you can load the saved model file (model.h5) and run predictions on new images.

---

### References
Keras Documentation: Keras.io

DeepFace Library: DeepFace Documentation

---

### Appendices
Presentation V10.pptx: Detailed project presentation

Notebook or Scripts: Jupyter notebooks and scripts used for training and evaluation
