from functions import load_fsdd
import pathlib
from utils import *
import wandb
from WANDB import config
from datetime import datetime
from VV_train import train
from VV_autoencoder import cond01, cond10
import numpy as np
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
path_features_matching_keyboard_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_strings/'

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


# ---------------------------------------------------------------------------------------------------------------------
# CONDITIONAL INPUTS
# ---------------------------------------------------------------------------------------------------------------------

def conditional_input(spectrograms_path, label):
    """

        :param spectrograms_path: where the normalised spectrograms are
        :return: x_train, array with all spectrograms data appended

        """
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        count = 0
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            spectrogram = spectrogram[0:512, 0:63]
            spectrogram_conc = np.concatenate((spectrogram, label), axis=1)
            x_train.append(spectrogram_conc)
            count += 1
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (4130, 512, 256, 1)
    return x_train



    """ Builds the conditional input and returns the original input images, their labels and the conditional input."""

    input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype='float32')(inputs[0])
    input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype='float32')(inputs[1])
    labels = tf.reshape(inputs[1], [-1, 1, 1, self.label_dim])  # batch_size, 1, 1, label_size
    ones = tf.ones([inputs[0].shape[0]] + self.image_dim[0:-1] + [self.label_dim])  # batch_size, 64, 64, label_size
    labels = ones * labels  # batch_size, 64, 64, label_size
    conditional_input = tf.keras.layers.InputLayer(
        input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype='float32')(
        tf.concat([inputs[0], labels], axis=3))

    return input_img, input_label, conditional_input


# ---------------------------------------------------------------------------------------------------------------------
# RUN THE MAIN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare CVAE')

    # LOAD X_TRAIN, Y_TRAIN, X_VAL AND Y_VAL
    x_train0 = np.squeeze(load_fsdd(x_train_SPECTROGRAMS_PATH))
    x_train0 = np.concatenate((x_train0, cond01), axis=1)
    x_train1 = np.squeeze(load_fsdd(x_train_SPECTROGRAMS_PATH))
    x_train1 = np.concatenate((x_train1, cond10), axis=1)
    x_train = [x_train0, x_train1]

    y_train0 = load_fsdd(y_train0_SPECTROGRAMS_PATH)
    y_train1 = load_fsdd(y_train1_SPECTROGRAMS_PATH)
    y_train = [y_train0, y_train1]

    x_val0 = np.squeeze(load_fsdd(x_val_SPECTROGRAMS_PATH))
    x_val0 = np.concatenate((x_val0, cond01), axis=1)
    x_val1 = np.squeeze(load_fsdd(x_val_SPECTROGRAMS_PATH))
    x_val1 = np.concat((x_val1, cond10), axis=1)
    x_val = [x_val0, x_val1]

    y_val0 = load_fsdd(y_val0_SPECTROGRAMS_PATH)
    y_val1 = load_fsdd(y_val1_SPECTROGRAMS_PATH)
    y_val = [y_val0, y_val1]

    x_train = np.expand_dims(x_train, axis=(0, -1))
    # y_train = np.expand_dims(y_train, axis=(0, -1))
    x_val = np.expand_dims(x_val, axis=(0, -1))
    # y_val = np.expand_dims(y_val, axis=(0, -1))

    # PRINT SHAPES
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('y_val.shape: ', y_val.shape)

    # TRAIN MODEL
    autoencoder = train(x_train, y_train, x_val, y_val, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/CVAE/" + dt_string)

