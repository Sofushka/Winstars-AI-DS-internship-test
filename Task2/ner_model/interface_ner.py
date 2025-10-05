import spacy


class NERInterface():
    """
    Class for loading and using a trained NER model    
    """
    def __init__(self, model_path = 'ner_model/text_model'):
        """
        Load the trained NER model

        :param model_path: Path to the trained model
        """
        self.model = spacy.load(model_path)

    def predict(self, text):
        """
        Take text and make predictions with the model  

        :param text: Text for model prediction
        :return: List of entities found in text (e.g., animal names)
        """    
        if not text:
            return []
        doc = self.model(text)
        return [ent.text for ent in doc.ents]