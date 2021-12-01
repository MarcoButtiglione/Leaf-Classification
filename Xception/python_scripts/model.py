import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers import NoiseLayer

# FIRST SIMPLE MODEL
def simple_model(input_shape, num_classes):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

# TRANSFER LEARNING WITH XCEPTION
def transfer_learning_model_Xception(input_shape, num_classes):
    base_model = keras.applications.Xception(weights='imagenet', input_shape=input_shape, include_top=False)
    # Freeze base model
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, 'constant'),
        layers.RandomZoom([-0.2, 0.2], fill_mode='constant'),
        layers.RandomTranslation([-0.1, 0.1], [-0.1, 0.1], 'constant')
    ])

    inputs = keras.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = data_augmentation(x)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model

# TRANSFER LEARNING WITH RESNET50V2
def transfer_learning_model_ResNet50V2(input_shape, num_classes):
    base_model = keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)
    # Freeze base model
    base_model.trainable = False

    seed = 0

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.5, 'constant', seed=seed),
        layers.RandomZoom([-0.2, 0.2], fill_mode='constant', seed=seed),
        layers.RandomTranslation([-0.1, 0.1], [-0.1, 0.1], 'constant', seed=seed),
        #NoiseLayer(256)
    ])

    inputs = keras.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = data_augmentation(x)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5, seed=seed)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model