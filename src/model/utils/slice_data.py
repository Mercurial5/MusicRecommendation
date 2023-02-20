import os
import re
from PIL import Image

"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""


def slice_train_spectrogram():
    sliced_images_path = 'train_sliced_images'
    if os.path.exists(sliced_images_path):
        return

    images_path = 'train_spectrogram_images'
    filenames = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".jpg")]

    print('Slicing Spectrograms')

    if not os.path.exists(sliced_images_path):
        os.makedirs(sliced_images_path)

    counter = 0
    for file in filenames:
        try:
            genre_variable = re.search(fr'{images_path}\\.*_(.+?).jpg', file).group(1)
        except AttributeError:
            print(fr'{images_path}\\.*_(.+?).jpg', file)
            exit(1)
        img = Image.open(file)
        subsample_size = 128
        width, height = img.size
        number_of_samples = width // subsample_size
        for i in range(number_of_samples):
            start = i * subsample_size
            img_temporary = img.crop((start, 0, start + subsample_size, subsample_size))
            img_temporary.save(f'{sliced_images_path}/{counter}_{genre_variable}.jpg')
            counter += 1


def slice_test_spectrogram():
    sliced_images_path = 'test_sliced_images'
    if os.path.exists(sliced_images_path):
        return

    images_path = 'test_spectrogram_images'
    filenames = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".jpg")]

    print('Slicing Spectrograms')
    if not os.path.exists(sliced_images_path):
        os.makedirs(sliced_images_path)

    counter = 0
    for file in filenames:
        song_variable = re.search(fr'{images_path}\\(.+?).jpg', file).group(1)
        img = Image.open(file)
        subsample_size = 128
        width, height = img.size
        number_of_samples = width // subsample_size
        for i in range(number_of_samples):
            start = i * subsample_size
            img_temporary = img.crop((start, 0, start + subsample_size, subsample_size))
            img_temporary.save(f"{sliced_images_path}/" + str(counter) + "_" + song_variable + ".jpg")
            counter = counter + 1


def slice_spectrogram(mode: str):
    if mode == 'Train':
        slice_train_spectrogram()
    elif mode == 'Test':
        slice_test_spectrogram()
