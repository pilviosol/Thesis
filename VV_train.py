import os

import numpy as np
import pathlib
from VV_autoencoder import VAE
from utils import *
from NSYNTH_feature_extractor import path_features_matching_flute, path_features_matching_vocal

set_gpu(-1)

LEARNING_RATE = 0.0005
BATCH_SIZE = 16
EPOCHS = 150

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute)
x_train_SPECTROGRAMS_PATH_dir = x_train_SPECTROGRAMS_PATH.iterdir()

y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal)
y_train_SPECTROGRAMS_PATH_dir = y_train_SPECTROGRAMS_PATH.iterdir()


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            spectrogram = spectrogram[0:512, 0:256]
            print("file_name: ", file_name)
            print("shape: ", spectrogram.shape)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train


def train(x_train, y_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(512, 256, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    print('ollare')
    x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    y_train = load_fsdd(y_train_SPECTROGRAMS_PATH)
    autoencoder = train(x_train, y_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("VV_model")
