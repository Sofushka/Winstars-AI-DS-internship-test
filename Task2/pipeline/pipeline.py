from ner_model.interface_ner import NERInterface
from image_classification_model.inference_image_model import ImageClassifierInterface
import tensorflow as tf


class PipelineExecution():
    """
    Class to combine NER and Image Classification models
    to decide if the text matches the animal in the image
    """
    def __init__(self, train_data_dir = "data/train"):
        """
        Load trained NER and Image Classification models,
        and store class names for image classification mapping
        """
        self.ner = NERInterface()
        self.img_cl = ImageClassifierInterface()
        self.class_names = self._load_class_names(train_data_dir)

    def _load_class_names(self, data_dir):
        """
        Load dataset to get the class names in correct order.

        :param data_dir: path to training dataset
        :return: list of class names
        """
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(160, 160),
            batch_size=16
        )
        return dataset.class_names

    def is_true(self, text = 'There is a cow in the picture.', img_path = 'data/test/cow/OIP-__wtYf6sbvlOlMBMHyA3OAAAAA.jpeg'):
        """
        Decide if the text matches the animal in the image

        :param text: Text to analyze
        :param img_path: Path to image to analyze
        :return: 1 if text matches image, 0 otherwise
        """
        #Get predictions from both models
        ner_pred = self.ner.predict(text)
        img_pred_idex = self.img_cl.predict(img_path)     

        #Convert prediction of NER into text
        img_pred_class = self.class_names[img_pred_idex]            

        # Normalize text: convert to lowercase
        ner_text = ner_pred[0].lower() if ner_pred else ""
        img_text = img_pred_class.lower()

        #Compare results, handle simple plural forms
        if ner_text == img_text or ner_text == img_text + "s" or ner_text == 'butterflies':
            return 1
        else:
            return 0
        

