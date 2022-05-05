from functions import load_fsdd_concat, load_fsdd
import pathlib
from utils import *
import wandb
from WANDB import config
from datetime import datetime
from VV_train import train
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import os

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

path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/'
path_features_matching_string_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_strings/'
path_features_matching_keyboard_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_keyboards/'

path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"
path_features_matching_string_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_string_0305_VALID/"
path_features_matching_keyboard_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_keyboard_0305_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
y_train0_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_string_train)
y_train1_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_keyboard_train)

x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)
y_val0_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_string_val)
y_val1_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_keyboard_val)


LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LATENT_DIM = config['latent_dim']
INPUT_SHAPE = config['input_shape']
CONV_FILTERS = config['conv_filters']
CONV_KERNELS = config['conv_kernels']
CONV_STRIDES = config['conv_strides']

ones = np.ones([512, 64], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones = np.repeat(ones, 824, axis=0)

zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros = np.repeat(zeros, 824, axis=0)

cond_enc = np.concatenate((ones, zeros), axis=0)

cond01 = np.asarray([0, 1])
cond01 = np.expand_dims(cond01, axis=0)
# cond01 = np.repeat(cond01, BATCH_SIZE, axis=0)

cond10 = np.asarray([1, 0])
cond10 = np.expand_dims(cond10, axis=0)
# cond01 = np.repeat(cond01, BATCH_SIZE, axis=0)

cond_dec = np.concatenate((cond01, cond10), axis=0)
# ---------------------------------------------------------------------------------------------------------------------
# RUN THE MAIN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare CVAE')

    # LOAD X_TRAIN, Y_TRAIN, X_VAL AND Y_VAL
    x_train0 = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train1 = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train = np.concatenate((x_train0, x_train1), axis=0)
    x_train = [x_train, cond_enc, cond_dec]

    y_train0 = load_fsdd(y_train0_SPECTROGRAMS_PATH)
    y_train1 = load_fsdd(y_train1_SPECTROGRAMS_PATH)
    y_train = np.concatenate((y_train0, y_train1), axis=0)


    x_val0 = load_fsdd_concat(x_val_SPECTROGRAMS_PATH, 0)
    x_val1 = load_fsdd_concat(x_val_SPECTROGRAMS_PATH, 1)
    x_val = np.concatenate((x_val0, x_val1), axis=0)


    y_val0 = load_fsdd(y_val0_SPECTROGRAMS_PATH)
    y_val1 = load_fsdd(y_val1_SPECTROGRAMS_PATH)
    y_val = np.concatenate((y_val0, y_val1), axis=0)


    # PRINT SHAPES
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('y_val.shape: ', y_val.shape)


    # TRAIN MODEL
    autoencoder = train(x_train, y_train, x_val, y_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/CVAE/" + dt_string)

