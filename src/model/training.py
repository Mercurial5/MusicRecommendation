import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.initializers.initializers_v1 import HeNormal

from model.utils import load_dataset


def train():
    train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(mode="Train", dataset_size=1)
    print(len(train_x), len(train_y), len(test_x), len(test_y))

    # Expand the dimensions of the image to have a channel dimension. (nx128x128) ==> (nx128x128x1)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

    # Normalize the matrices.
    train_x = train_x / 255.
    test_x = test_x / 255.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (7, 7), kernel_initializer=HeNormal(seed=1), activation='relu',
                               input_shape=(128, 128, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(128, (7, 7), kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2, 2), strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.Dense(256, kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.Dense(64, kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.Dense(32, kernel_initializer=HeNormal(seed=1), activation='relu'),
        tf.keras.layers.Dense(n_classes, kernel_initializer=HeNormal(seed=1), activation='softmax')
    ])

    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    pd.DataFrame(model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1).history).to_csv(
        "Saved_Model/training_history.csv")
    score = model.evaluate(test_x, test_y, verbose=1)
    print(score)
    model.save("Saved_Model/Model.h5")
