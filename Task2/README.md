# Task 2: Named entity recognition + image classification

## Project Overview
This project implements image classification on the **animals10** dataset using **MobileNetV2** 
and Named Entity Recognition (NER) on text using **en_core_web_md**.  

A **pipeline** combines both models to decide whether the text matches the animal in the image.

---

## Folder Structure

- `requirements.txt`  
  Contains all libraries used in the project

- `data/` 
  Contains training, validation and test data for image classification model
  -`test/` 
    Test data
  -`train/` 
    Train data
  -`val/` 
    Validation data 

- `image_classification_model/`
  Contains the implementations of image classification model and interface:

  -`saved_model/`
    Folder with trained image classification model

    -`animals10_model.keras`
      Trained image classification mode

  -`inference_image_model.py`
    Image classification interface   
    Methods:      
    - `predict(img_path)`  
    Input: path to image for prediction

  -`train_image_model.py`
    Image classification model training using MobileNetV2 and animals10 dataset

- `ner_model/`
  Contains the implementations of NER model and interface:

  -`text_model/`
    Folder with trained NER model    

  -`interface_ner.py`
    NER model interface  
    Methods:      
    - `predict(text)`  
    Input: text for prediction

  -`train_ner.py`
    NER model training using `en_core_web_md` and text data

  -`ner_train_data.py`
    Generated training dataset for NER

- `pipeline/`
  Combines NER and Image Classification models in one class `PipelineExecution`
  Methods:      
  - `_load_class_names(data_dir)`    
  -`is_true(text, img_path)`
  Input: text and image path, returns 1 if matches, otherwise 0

- `demo.ipynb`  
   Jupyter Notebook demonstrating usage of all models:
  - EDA of `animals10` dataset
  - Examples of how the solution works
  - Edge cases and input requirements

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
4. **Verify your data structure**:
-Ensure data/train, data/val, and data/test folders exist for the image classification dataset.
-Make sure ner_model/ner_train_data.py contains your NER training data.

5. **Run demo Notebook** to see examples of how the solution works and to explore edge cases.

6. **Run the pipeline script** to test the full pipeline with text and image inputs:
    ```bash
    pip install -r requirements.txt
    ```







