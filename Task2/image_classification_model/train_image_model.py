import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN optimizations


import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import MobileNetV2


# Load dataset and resize it to the input shape expected by the model
train_dt = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(160, 160),
    batch_size=16
)

val_dt = tf.keras.utils.image_dataset_from_directory(
    "data/val",
    image_size=(160, 160),
    batch_size=16
)

test_dt = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(160, 160),
    batch_size=16
)

# Preprocess (normalize images to [-1,1] for MobileNetV2)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
train_dt = train_dt.map(lambda x, y: (preprocess_input(x), y))
val_dt = val_dt.map(lambda x, y: (preprocess_input(x), y))
test_dt = test_dt.map(lambda x, y: (preprocess_input(x), y))

# Cache data in memory, shuffle for training randomness, prefetch to overlap I/O and training
autotune = tf.data.AUTOTUNE
train_dt = train_dt.cache().shuffle(1000, seed=54).prefetch(buffer_size=autotune)
val_dt = val_dt.cache().prefetch(buffer_size=autotune)
test_dt = test_dt.cache().prefetch(buffer_size=autotune)

# Build model
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_dt, validation_data=val_dt, epochs=5)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_dt)
print("Test accuracy:", test_acc)

# Save trained model
model.save("image_classification_model/saved_model/animals10_model.keras")
