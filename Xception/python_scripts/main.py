import os
import random

import numpy as np
from sklearn.utils import class_weight
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import data
from model import simple_model, transfer_learning_model_Xception, transfer_learning_model_ResNet50V2

datasetDir = 'training'
classes = sorted(os.listdir(datasetDir))

input_shape = (256, 256, 3)
batch_size = 8
epochs = 200

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# LOAD DATA
(X_train, y_train), (X_val, y_val) = data.load_splitted_data_without_test(classes, datasetDir)

# CREATE THE MODEL
#model = simple_model(input_shape=input_shape, num_classes=len(classes))
#model = transfer_learning_model_Xception(input_shape=input_shape, num_classes=len(classes))
model = transfer_learning_model_ResNet50V2(input_shape=input_shape, num_classes=len(classes))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"], run_eagerly=False)

class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train.argmax(axis=1)), y_train.argmax(axis=1))))

# CALLBACKS
monitor = 'val_accuracy'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join('ckpt', 'best_weights.hdf5'), monitor=monitor, verbose=1, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch'
    ),
    # tf.keras.callbacks.EarlyStopping(
    #     monitor=monitor, min_delta=0, patience=0, verbose=0,
    #     mode='max', baseline=1, restore_best_weights=True
    # )
    tf.keras.callbacks.TensorBoard(
        log_dir='log', update_freq='epoch'
    )
]

# FIT THE MODEL
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=callbacks)

del X_train

print("loading best weights")
model.load_weights(os.path.join('ckpt', 'best_weights.hdf5'))

# METRICS
print("calculating metrics on test set")
score = model.evaluate(X_val, y_val, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
#print("Test Macro-F1:", score[2].mean())
print("Test F1:", score[2])

print("calculating predictions on test set")
y_pred = model.predict(X_val, batch_size=batch_size)

# CONFUSION MATRIX
print("calculating confusion matrix")
cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
ncm = cm / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(confusion_matrix=ncm, display_labels=range(len(classes)))
disp.plot()
plt.show()








# TODO: L2 penalty
# TODO: cambiare rete? resnet50
# TODO: mettere data augmentation con ImageGenerator

