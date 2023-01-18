# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    val_size = 0.4
    training_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_size
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator= training_datagen.flow_from_directory(TRAINING_DIR,
                                                    subset="training",
                                                    batch_size = 16,
                                                    class_mode = 'categorical',
                                                    target_size = (150, 150))# YOUR CODE HERE
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255,validation_split=val_size)
    validation_generator =  validation_datagen.flow_from_directory( TRAINING_DIR,
                                                    subset="validation",
                                                    batch_size  = 16,
                                                    class_mode  = 'categorical',
                                                    target_size = (150, 150))
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    batch_size=16
    model.compile(optimizer=Adam(lr=0.00146), loss='categorical_crossentropy', metrics=['accuracy'])
    validation_steps = validation_generator.samples / validation_generator.batch_size - 1
    model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=50,
        epochs=25,
        validation_steps=validation_steps,
        verbose=2
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
