import spacy
from spacy.util import minibatch
from spacy.training.example import Example
from ner_train_data import train_data

import random

#Load pre-trained English model (medium size)
nlp = spacy.load('en_core_web_md')

#Check if the NER component is in the pipeline, if not, add it
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

# Add the label "ANIMAL" to NER
for _, annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

#Disable other pipeline components while training NER to save time
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    # Initialize the model's optimizer
    optimizer = nlp.begin_training()

    epochs = 20
    for epoch in range(epochs):
        random.shuffle(train_data)  # shuffle data every epoch
        losses = {}
        # Create small batches to save memory
        batches = minibatch(train_data, size=3)
        for batch in batches:
            examples = []
            # Convert each text + annotation into Example objects
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            # Update the NER model using this batch
            nlp.update(examples, drop=0.5, losses=losses)

        print(f"Epoch {epoch+1} Losses {losses},")

#Save trained NER model to disk 
nlp.to_disk(path="ner_model/text_model")



