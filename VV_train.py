import os

import numpy as np
import pathlib
from VV_autoencoder import VAE
from utils import *

set_gpu(-1)

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 150

SPECTROGRAMS_PATH = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train_resampled_256/')
SPECTROGRAMS_PATH_dir = SPECTROGRAMS_PATH.iterdir()


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            # spectrogram = spectrogram[0:1024, 0:128]
            print("file_name: ", file_name)
            print("shape: ", spectrogram.shape)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(512, 256, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    print('ollare')
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("VV_model")
