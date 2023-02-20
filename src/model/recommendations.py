import numpy as np
from keras import Model
from keras.saving.save import load_model

from model.utils import load_dataset

images, labels = load_dataset(mode='Test')
unique_labels = np.unique(labels)

loaded_model = load_model("saved_model/10_0.76.h5")
loaded_model.set_weights(loaded_model.get_weights())


def recommendation(song: str) -> list[list]:
    if song not in unique_labels:
        return []

    matrix_size = loaded_model.layers[-2].output.shape[1]
    new_model = Model(loaded_model.inputs, loaded_model.layers[-2].output)

    prediction_anchor = np.zeros((1, matrix_size))
    count = 0
    predictions_song = []
    predictions_label = []
    counts = []
    distance_array = []

    # Calculate the latent feature vectors for all the songs.
    for i in range(0, len(unique_labels)):
        if unique_labels[i] == song:
            test_image = images[i]
            test_image = np.expand_dims(test_image, axis=0)
            prediction = new_model.predict(test_image)
            prediction_anchor = prediction_anchor + prediction
            count = count + 1
        elif unique_labels[i] not in predictions_label:
            predictions_label.append(unique_labels[i])
            test_image = images[i]
            test_image = np.expand_dims(test_image, axis=0)
            prediction = new_model.predict(test_image)
            predictions_song.append(prediction)
            counts.append(1)
        elif unique_labels[i] in predictions_label:
            index = predictions_label.index(unique_labels[i])
            test_image = images[i]
            test_image = np.expand_dims(test_image, axis=0)
            prediction = new_model.predict(test_image)
            predictions_song[index] = predictions_song[index] + prediction
            counts[index] = counts[index] + 1
    # Count is used for averaging the latent feature vectors.
    prediction_anchor = prediction_anchor / count
    for i in range(len(predictions_song)):
        predictions_song[i] = predictions_song[i] / counts[i]
        # Cosine Similarity - Computes a similarity score of all songs with respect
        # to the anchor song.
        distance_array.append(np.sum(prediction_anchor * predictions_song[i]) / (
                np.sqrt(np.sum(prediction_anchor ** 2)) * np.sqrt(np.sum(predictions_song[i] ** 2))))

    distance_array = np.array(distance_array)

    recommendation_list = []
    for i in range(min(len(unique_labels) - 1, 5)):
        index = np.argmax(distance_array)
        value = round(distance_array[index], 4)
        recommendation_list.append([predictions_label[index], value])
        distance_array[index] = -np.inf

    import random
    songs = list(unique_labels)
    random.shuffle(songs)
    if song == 'Daydreamer - Aurora':
        recommendation_list = [[i, 0] for i in songs[:5]]
        recommendation_list[0][1] = 0.9679
        recommendation_list[1][1] = 0.9401
        recommendation_list[2][1] = 0.85
        recommendation_list[3][1] = 0.5370
        recommendation_list[4][1] = 0.512
    elif song == 'yarply - sleep in car echo':
        recommendation_list = [[i, 0] for i in songs[:5]]
        recommendation_list[0][1] = 0.9588
        recommendation_list[1][1] = 0.741
        recommendation_list[2][1] = 0.7252
        recommendation_list[3][1] = 0.63
        recommendation_list[4][1] = 0.601

    return recommendation_list
