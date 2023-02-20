import os
import re
import numpy as np
import cv2
from model.utils import create_spectrogram
from model.utils import slice_spectrogram
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

"""
Converts images and labels into training and testing matrices.
"""


def load_train_dataset(dateset_size: float):
    genre = {"Hip-Hop": 0, "International": 1, "Electronic": 2, "Folk": 3, "Experimental": 4, "Rock": 5, "Pop": 6,
             "Instrumental": 7}

    n_classes = len(genre)
    genre_new = {value: key for key, value in genre.items()}

    if os.path.exists('Training_Data'):
        train_x = np.load("Training_Data/train_x.npy")
        train_y = np.load("Training_Data/train_y.npy")
        test_x = np.load("Training_Data/test_x.npy")
        test_y = np.load("Training_Data/test_y.npy")
        return train_x, train_y, test_x, test_y, n_classes, genre_new

    print('Compiling Training set')

    sliced_images_path = 'train_sliced_images'

    filenames = [os.path.join(sliced_images_path, f) for f in os.listdir(sliced_images_path) if f.endswith(".jpg")]
    images_all: list[int | None] = [None] * (len(filenames))
    labels_all: list[int | None] = [None] * (len(filenames))

    for f in filenames:
        index = int(re.search(fr'{sliced_images_path}\\(.+?)_.*.jpg', f).group(1))
        genre_variable = re.search(fr'{sliced_images_path}\\.*_(.+?).jpg', f).group(1)
        temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        labels_all[index] = genre[genre_variable]

    if dateset_size == 1.0:
        images = images_all
        labels = labels_all
    else:
        count_max = int(len(images_all) * dateset_size / 8.0)
        count_array = [0, 0, 0, 0, 0, 0, 0, 0]
        images = []
        labels = []

        for index, _ in enumerate(images_all):
            if count_array[labels_all[index]] > count_max:
                continue

            images.append(images_all[index])
            labels.append(labels_all[index])
            count_array[labels_all[index]] += 1

    images = np.array(images)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], 1)
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.05, shuffle=True)

    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y, num_classes=8)

    os.makedirs('Training_Data')
    np.save("Training_Data/train_x.npy", train_x)
    np.save("Training_Data/train_y.npy", train_y)
    np.save("Training_Data/test_x.npy", test_x)
    np.save("Training_Data/test_y.npy", test_y)
    return train_x, train_y, test_x, test_y, n_classes, genre_new


def load_test_dataset():
    print('Compiling Training set')
    sliced_images_path = 'test_sliced_images'

    filenames = [os.path.join(sliced_images_path, f) for f in os.listdir(sliced_images_path) if f.endswith(".jpg")]
    images = []
    labels = []
    for f in filenames:
        song_variable = re.search(fr'{sliced_images_path}\\.*_(.+?).jpg', f).group(1)
        temp_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        images.append(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY))
        labels.append(song_variable)

    images = np.array(images)

    return images, labels


def load_dataset(mode: str = None, dataset_size: float = 1.0):
    create_spectrogram(mode)
    slice_spectrogram(mode)

    # datasetSize is a float value which returns a fraction of the dataset.
    # If set as 1.0 it returns the entire dataset.
    # If set as 0.5 it returns half the dataset.
    if mode == 'Train':
        return load_train_dataset(dataset_size)
    elif mode == 'Test':
        return load_test_dataset()
