import os

import numpy as np
from sklearn.utils import class_weight
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import data
from model import simple_model, transfer_learning_model_Xception

datasetDir = 'training'
classes = sorted(os.listdir(datasetDir))

input_shape = (256, 256, 3)
batch_size = 8
epochs = 100

# load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data.load_splitted_data(classes, datasetDir)

# create the model
#model = simple_model(input_shape=input_shape, num_classes=len(classes))
model = transfer_learning_model_Xception(input_shape=input_shape, num_classes=len(classes))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", F1Score(num_classes=14)], run_eagerly=True)

del X_train
del X_val

print("loading best weights")
model.load_weights(os.path.join('ckpt', 'best_weights.hdf5'))

print("calculating metrics on test set")
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test Macro-F1:", score[2].mean())
print("Test F1:", score[2])

print("calculating predictions on test set")
y_pred = model.predict(X_test, batch_size=batch_size)

print("calculating confusion matrix")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
ncm = cm / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(confusion_matrix=ncm, display_labels=range(len(classes)))
disp.plot()
plt.show()
