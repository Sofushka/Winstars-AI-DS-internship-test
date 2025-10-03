# Task 1: Image Classification + OOP

## Project Overview
This project implements image classification on the MNIST dataset using three different models:
- Random Forest
- Feed-Forward Neural Network (FNN)
- Convolutional Neural Network (CNN)

All models are implemented with a **common interface** using an abstract class `MnistClassifierInterface`.  
A wrapper class `MnistClassifier` provides a unified way to select and use any of the three models without changing the input/output structure.

---

## Folder Structure

- `requirements.txt`  
  Contains all libraries used in the project.

- `classifiers/`  
  Contains the implementations of models and interface:

  - `interface.py`  
    Abstract class `MnistClassifierInterface` defines the common interface with `train` and `predict` methods.  
    All models implement this interface.

  - `cnn.py`  
    Convolutional Neural Network (CNN) implementation using `MnistClassifierInterface`.  
    Methods:
    - `train(X, y, epochs=5, batch_size=64)`  
    - `predict(X)`  
    Input for CNN must have shape `(num_samples, 28, 28, 1)`.

  - `nn.py`  
    Feed-Forward Neural Network (FNN) implementation using `MnistClassifierInterface`.  
    Methods:
    - `train(X, y, epochs=5, batch_size=64)`  
    - `predict(X)`  
    Input shape: `(num_samples, 28, 28)`.

  - `rf.py`  
    Random Forest implementation using `MnistClassifierInterface`.  
    Methods:
    - `train(X, y)`  
    - `predict(X)`  
    Input shape: `(num_samples, 28*28)` or flattened.

  - `mnist_classifier.py`  
    Wrapper class to select the algorithm (`'rf'`, `'nn'`, `'cnn'`) and provide predictions with a unified interface.

- `demo.ipynb`  
  Jupyter Notebook demonstrating usage of all models, including:
  - Loading and preprocessing MNIST dataset  
  - Training and predicting with each model  
  - Examples of edge cases and input shape requirements

---

## Setup Instructions

1. **Install Python 3.10+** if not already installed.

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Linux/Mac
   ```
3. **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Run demo Notebook** to see examples of how the solution works and to explore edge cases.