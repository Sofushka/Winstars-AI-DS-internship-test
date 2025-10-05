import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


class ImageClassifierInterface():
    """
    Class for loading and using a trained image classification model
    """
    def __init__(self, model_path='image_classification_model/saved_model/animals10_model.keras'):
        """
        Load the trained image classification model

        :param model_path: Path to the trained model
        """
        self.model = tf.keras.models.load_model(model_path)
    

    def predict(self, img_path):
        """
         Load an image, preprocess it, 
         and make predictions with the model 

        :param img_path: Path to the image file
        :return: (predicted_class_index, confidence_score)
        """      
        #Load the image and resize it to the input shape expected by the model  
        img = image.load_img(img_path, target_size=(160, 160))
        #Convert the image to a numpy array
        X = image.img_to_array(img)
        #Add a new dimension to match the model input shape
        X = np.expand_dims(X, axis=0)
        #Normalize the image to the range expected by MobileNetV2 ([-1, 1])
        X = preprocess_input(X)

        #Get predicted probabilities for each class
        pred = self.model.predict(X)   
        #Select the class with the highest probability
        class_idx = np.argmax(pred, axis=1)[0] 
        #Get the confidence score for the predicted class
        confidence = pred[0][class_idx]

        

        return class_idx


