from functions import load_fsdd
import pathlib
from utils import *
import wandb
from WANDB import config
from datetime import datetime
from VV_autoencoder import VAE
import numpy as np


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


# ENCODER CONDITIONING MATRICES
ones = np.ones([512, 64], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones_train = np.repeat(ones, 824, axis=0)
ones_val = np.repeat(ones, 46, axis=0)

dues_train = np.add(ones_train, ones_train)
dues_val = np.add(ones_val, ones_val)


zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros_train = np.repeat(zeros, 824, axis=0)
zeros_val = np.repeat(zeros, 46, axis=0)

cond_enc_train = np.concatenate((ones_train, zeros_train), axis=0)
cond_enc_val = np.concatenate((ones_val, zeros_val), axis=0)


'''
cond_enc_train = np.concatenate((ones_train, dues_train), axis=0)
cond_enc_val = np.concatenate((ones_val, dues_val), axis=0) '''

# DECODER CONDITIONING VECTORS
cond01 = np.asarray([0, 1])
cond01 = np.expand_dims(cond01, axis=0)
cond01_train = np.repeat(cond01, 824, axis=0)
cond01_val = np.repeat(cond01, 46, axis=0)

cond10 = np.asarray([1, 0])
cond10 = np.expand_dims(cond10, axis=0)
cond10_train = np.repeat(cond10, 824, axis=0)
cond10_val = np.repeat(cond10, 46, axis=0)

cond_dec_train = np.concatenate((cond01_train, cond10_train), axis=0)
cond_dec_val = np.concatenate((cond01_val, cond10_val), axis=0)


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
    print('ollare CVAE')



    # LOAD X_TRAIN, Y_TRAIN, X_VAL AND Y_VAL
    x_train0 = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train1 = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train = np.concatenate((x_train0, x_train1), axis=0)
    x_train = [x_train, cond_enc_train, cond_dec_train]

    y_train0 = load_fsdd(y_train0_SPECTROGRAMS_PATH)
    y_train1 = load_fsdd(y_train1_SPECTROGRAMS_PATH)
    y_train = np.concatenate((y_train0, y_train1), axis=0)

    x_val0 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val1 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val = np.concatenate((x_val0, x_val1), axis=0)
    x_val = [x_val, cond_enc_val, cond_dec_val]

    y_val0 = load_fsdd(y_val0_SPECTROGRAMS_PATH)
    y_val1 = load_fsdd(y_val1_SPECTROGRAMS_PATH)
    y_val = np.concatenate((y_val0, y_val1), axis=0)

    # PRINT SHAPES
    '''
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('y_val.shape: ', y_val.shape) '''

    # TRAIN MODEL
    autoencoder = train(x_train, y_train, x_val, y_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/CVAE/" + dt_string)
    encoded_x_val = autoencoder.tsne(x_val, perplexity=16, title='x_val', annotations=annotations, color='red')
