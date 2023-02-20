import os
import re
from datetime import datetime

import audioread
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

"""
Convert 30s mp3 files into mel-spectrograms.

A mel-spectrograms is a kind of time-frequency representation.
It is obtained from an audio signal by computing the Fourier transforms of short, overlapping windows.
Each of these Fourier transforms constitutes a frame.
These successive frames are then concatenated into a matrix to form the spectrogram.
"""


def create_train_spectrogram():
    spectrograms_path = 'train_spectrogram_images'
    if os.path.exists(spectrograms_path):
        return

    filename_metadata = "dataset/fma_metadata/tracks.csv"
    tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
    tracks_array = tracks.values
    tracks_id_array = tracks_array[:, 0]
    tracks_genre_array = tracks_array[:, 40]
    tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
    tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

    folder_sample = 'dataset/fma_small'
    directories = [d for d in os.listdir(folder_sample) if os.path.isdir(os.path.join(folder_sample, d))]

    if not os.path.exists(spectrograms_path):
        os.makedirs(spectrograms_path)

    print('Converting mp3 audio files into mel spectrograms', datetime.now())

    counter = 3897
    for d in directories[12:107]:
        label_directory = os.path.join(folder_sample, d)
        filenames = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".mp3")]

        for file in filenames:
            track_id = int(re.search(r'fma_small\\.*\\(.+?).mp3', file).group(1))
            track_index = list(tracks_id_array).index(track_id)
            if str(tracks_genre_array[track_index, 0]) != '0':
                if tracks_genre_array[track_index, 0] == 'Old-Time / Historic':
                    tracks_genre_array[track_index, 0] = 'Historic'
                path = f'{spectrograms_path}/{counter}_{tracks_genre_array[track_index, 0]}.jpg'
                if not os.path.exists(path):
                    _convert_mp3_to_spectrogram(file, path)
                else:
                    print(path)
                counter += 1


def create_test_spectrogram():
    spectrograms_path = 'test_spectrogram_images'
    if os.path.exists(spectrograms_path):
        return

    folder_sample = 'dataset/test_dataset'

    if not os.path.exists(spectrograms_path):
        os.makedirs(spectrograms_path)

    filenames = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample) if f.endswith(".mp3")]

    print("Converting mp3 audio files into mel Spectrograms")
    for file in filenames:
        test_id = re.search(fr'{folder_sample}\\(.+?).mp3', file).group(1)
        _convert_mp3_to_spectrogram(file, path=f'{spectrograms_path}/{test_id}.jpg')


def _convert_mp3_to_spectrogram(file: str, path: str):
    try:
        y, sr = librosa.load(file)
    except audioread.NoBackendError:
        print(f'File {file} is corrupt')
        return

    print(file, path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel = librosa.power_to_db(mel_spectrogram)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = float(mel.shape[1]) / float(100)
    fig_size[1] = float(mel.shape[0]) / float(100)
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis('off')
    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel, cmap='gray_r')
    plt.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close()


def create_spectrogram(mode: str = None):
    if mode == 'Train':
        create_train_spectrogram()
    elif mode == 'Test':
        create_test_spectrogram()
