import os
import numpy as np
import pathlib
from VV_autoencoder import VAE
from utils import *
import wandb
from VV_autoencoder import train_loss, train_kl_loss, train_reconstruction_loss
from WANDB import config
import matplotlib.pyplot as plt

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
print(dt_string)
file2 = open(r"/nas/home/spol/Thesis/last_date.txt", "w+")
file2.write(dt_string)
file2.close()

wandb.init(project="my-test-project", entity="pilviosol", name=dt_string, config=config)
set_gpu(-1)


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_flute/'
path_features_matching_vocal_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_vocal/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_VALID/"
path_features_matching_vocal_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_vocal_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)
y_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_val)

LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        count = 0
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            print("file_name: ", file_name)
            x_train.append(spectrogram)
            count += 1
            print('count: ', count)
            if count % 100 == 0:
                fig = plt.figure()
                img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                                 aspect='auto')
                plt.title(file_name)
                plt.colorbar()
                plt.show()
                plt.close()
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (4130, 512, 256, 1)
    return x_train


def train(x_train, y_train, x_val, y_val, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(512, 256, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=config['latent_dim']
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, x_val, y_val, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    print('ollare')
    x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train = np.delete(x_train, 2776, axis=0)
    # y_train = load_fsdd(y_train_SPECTROGRAMS_PATH)
    # y_train = np.delete(y_train, 2776, axis=0)
    x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
    # y_val = load_fsdd(y_val_SPECTROGRAMS_PATH)
    autoencoder = train(x_train, x_train, x_val, x_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/" + dt_string)
