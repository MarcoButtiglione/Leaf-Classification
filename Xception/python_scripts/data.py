
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from split_dataset import split_dataset, split_dataset_without_test

''' DATA ANALYSIS '''

# READ THE CLASSES
datasetDir = 'training'
dataset_file = 'dataset.npy'
classes = sorted(os.listdir(datasetDir))
print('Number of classes:', len(classes))  # 14

count = 0
frequencies = []
for c in classes:
    # in order to not use the / (that we can't use in windows) I join the 2 paths
    my_path = os.path.join(datasetDir, c)
    files = [f for f in os.listdir(my_path) if isfile(join(my_path, f))]
    print(c, len(files), str(round(100*len(files)/17728, 2)) + "%")
    count += len(files)

    frequencies.append(len(files))

print('Number of total examples:', count)  # 17728

# BAR CHART
plt.title('Classes distribution')
plt.bar(list(map(lambda x: x[:3], classes)), frequencies)
plt.show()

''' DATA PREPARATION '''


def load_data(classes, datasetDir):
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            X = np.load(f)
            y = np.load(f)
        return X, y
    X = []  # contains all examples
    y = []  # contains all labels
    for c in classes:
        my_path = os.path.join(datasetDir, c)
        for i, image_file in enumerate(os.listdir(my_path)):
            if isfile(join(my_path, image_file)):
                image = cv2.imread(os.path.join(my_path, image_file))
                if image.shape != (256, 256, 3):
                    # Resize 2 images with different shape: 32004.jpg Peach, 41285.jpg Tomato
                    image = cv2.resize(image, (256, 256))
                X.append(image)
                y.append(classes.index(c))
            # if i == 500:
            # break

    # convert lists to numpy arrays
    X = np.stack(X)
    y = np.array(y)

    with open(dataset_file, 'wb') as f:
        np.save(f, X)
        np.save(f, y)

    return X, y


def load_splitted_data(classes, datasetDir):
    X, y = load_data(classes, datasetDir)
    y = to_categorical(y, len(classes))

    # split the dataset: 64%, 16%, 20%
    return split_dataset(X, y)

def load_splitted_data_without_test(classes, datasetDir):
    X, y = load_data(classes, datasetDir)
    y = to_categorical(y, len(classes))

    # split the dataset: 85%, 15%
    return split_dataset_without_test(X, y)





