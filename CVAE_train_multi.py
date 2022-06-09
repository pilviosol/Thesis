from functions import load_fsdd
import pathlib
from utils import *
import wandb
from WANDB import config
from datetime import datetime
from CVAE_autoencoder_multi import CVAEMulti
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

path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/07062022/NORMALIZED_flute/'
path_features_matching_string_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/07062022/NORMALIZED_string/'
path_features_matching_keyboard_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/07062022/NORMALIZED_keyboard/'
path_features_matching_guitar_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/07062022/NORMALIZED_guitar/'
path_features_matching_organ_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/07062022/NORMALIZED_organ/'

path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/NORMALIZED_flute/"
path_features_matching_string_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/NORMALIZED_flute/"
path_features_matching_keyboard_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/NORMALIZED_flute/"
path_features_matching_guitar_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/NORMALIZED_flute/"
path_features_matching_organ_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/NORMALIZED_flute/"


x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)

y_train0_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_string_train)
y_train1_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_keyboard_train)
y_train2_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_guitar_train)
y_train3_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_organ_train)


x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)

y_val0_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_string_val)
y_val1_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_keyboard_val)
y_val2_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_guitar_val)
y_val3_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_organ_val)


LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LATENT_DIM = config['latent_dim']
INPUT_SHAPE = config['input_shape']
CONV_FILTERS = config['conv_filters']
CONV_KERNELS = config['conv_kernels']
CONV_STRIDES = config['conv_strides']


# ENCODER CONDITIONING MATRICES

zeros = np.zeros([512, 256], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros_train = np.repeat(zeros, 708, axis=0)
zeros_val = np.repeat(zeros, 88, axis=0)

ones = np.ones([512, 256], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones_train = np.repeat(ones, 708, axis=0)
ones_val = np.repeat(ones, 88, axis=0)

twos_train = np.add(ones_train, ones_train)
twos_val = np.add(ones_val, ones_val)

threes_train = np.add(ones_train, twos_train)
threes_val = np.add(ones_val, twos_val)


cond_enc_train = np.concatenate((ones_train, zeros_train, twos_train, threes_train), axis=0)
cond_enc_val = np.concatenate((ones_val, zeros_val, twos_val, threes_val), axis=0)


# DECODER CONDITIONING VECTORS
cond0001 = np.asarray([0, 0, 0, 1])
cond0001 = np.expand_dims(cond0001, axis=0)
cond0001_train = np.repeat(cond0001, 708, axis=0)
cond0001_val = np.repeat(cond0001, 88, axis=0)

cond0010 = np.asarray([0, 0, 1, 0])
cond0010 = np.expand_dims(cond0010, axis=0)
cond0010_train = np.repeat(cond0010, 708, axis=0)
cond0010_val = np.repeat(cond0010, 88, axis=0)

cond0100 = np.asarray([0, 1, 0, 0])
cond0100 = np.expand_dims(cond0100, axis=0)
cond0100_train = np.repeat(cond0100, 708, axis=0)
cond0100_val = np.repeat(cond0100, 88, axis=0)

cond1000 = np.asarray([1, 0, 0, 0])
cond1000 = np.expand_dims(cond1000, axis=0)
cond1000_train = np.repeat(cond1000, 708, axis=0)
cond1000_val = np.repeat(cond1000, 88, axis=0)


cond_dec_train = np.concatenate((cond0001_train, cond0010_train, cond0100_train, cond1000_train), axis=0)
cond_dec_val = np.concatenate((cond0001_val, cond0010_val, cond0100_val, cond1000_val), axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# DEFINE TRAIN FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


def train(x_train, y_train, x_val, y_val, learning_rate, batch_size, epochs):
    autoencoder = CVAEMulti(
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
    print('ollare CVAE_multi')

    # LOAD X_TRAIN, Y_TRAIN, X_VAL AND Y_VAL
    x_train0 = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train = np.concatenate((x_train0, x_train0, x_train0, x_train0), axis=0)
    x_train = [x_train, cond_enc_train, cond_dec_train]

    y_train0 = load_fsdd(y_train0_SPECTROGRAMS_PATH)
    y_train1 = load_fsdd(y_train1_SPECTROGRAMS_PATH)
    y_train2 = load_fsdd(y_train2_SPECTROGRAMS_PATH)
    y_train3 = load_fsdd(y_train3_SPECTROGRAMS_PATH)
    y_train = np.concatenate((y_train0, y_train1, y_train2, y_train3), axis=0)

    x_val0 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val = np.concatenate((x_val0, x_val0, x_val0, x_val0), axis=0)
    x_val = [x_val, cond_enc_val, cond_dec_val]

    y_val0 = load_fsdd(y_val0_SPECTROGRAMS_PATH)
    y_val1 = load_fsdd(y_val1_SPECTROGRAMS_PATH)
    y_val2 = load_fsdd(y_val2_SPECTROGRAMS_PATH)
    y_val3 = load_fsdd(y_val3_SPECTROGRAMS_PATH)
    y_val = np.concatenate((y_val0, y_val1, y_val2, y_val3), axis=0)

    # TRAIN MODEL
    autoencoder = train(x_train, y_train, x_val, y_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/CVAE_multi/" + dt_string)
