import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


''' ADDING A NOISE LAYER TO IMMAGES '''

class NoiseLayer(keras.layers.Layer):

    def __init__(self, max_val):
        super(NoiseLayer, self).__init__()
        self.max_val = max_val

    def call(self, X, training=None):
        if X.shape[0] is not None:
            if training:
                if tf.experimental.numpy.random.randint(0,2):
                    noise = tf.random.uniform(shape=X.shape, maxval=self.max_val, dtype=tf.float32)
                    black = tf.reduce_sum(X, axis=3) == 0
                    noise = noise * tf.cast(tf.expand_dims(black, 3), tf.float32)
                    X = X + noise
        return X












