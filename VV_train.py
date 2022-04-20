from functions import load_fsdd
import pathlib
from VV_autoencoder import VAE
from utils import *
import wandb
from WANDB import config
from datetime import datetime


# ---------------------------------------------------------------------------------------------------------------------
# SET UP WANDB, SELECT GPU AND INSTANTIATING DATE AND TIME TO SAVE MODEL WITH THAT NAME
# ---------------------------------------------------------------------------------------------------------------------

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
print(dt_string)
file2 = open(r"/nas/home/spol/Thesis/last_date.txt", "w+")
file2.write(dt_string)
file2.close()

wandb.init(project="my-test-project", entity="pilviosol", name=dt_string, config=config)
set_gpu(-1)

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------

path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_FW_normalised_flute_TRAIN/'
path_features_matching_vocal_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_FW_normalised_string_TRAIN/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_VALID/"
path_features_matching_vocal_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_string_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)
y_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_val)

LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LATENT_DIM = config['latent_dim']
INPUT_SHAPE = config['input_shape']
CONV_FILTERS = config['conv_filters']
CONV_KERNELS = config['conv_kernels']
CONV_STRIDES = config['conv_strides']

# ---------------------------------------------------------------------------------------------------------------------
# DEFINE TRAIN FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


def train(x_train, y_train, x_val, y_val, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=INPUT_SHAPE,
        conv_filters=CONV_FILTERS,
        conv_kernels=CONV_KERNELS,
        conv_strides=CONV_STRIDES,
        latent_space_dim=LATENT_DIM
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, x_val, y_val, batch_size, epochs)
    return autoencoder


# ---------------------------------------------------------------------------------------------------------------------
# RUN THE MAIN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare')
    x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    y_train = load_fsdd(y_train_SPECTROGRAMS_PATH)
    x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
    y_val = load_fsdd(y_val_SPECTROGRAMS_PATH)
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('y_val.shape: ', y_val.shape)
    autoencoder = train(x_train, y_train, x_val, y_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/" + dt_string)
