import pathlib
from VV_autoencoder import VAE
from utils import *
import wandb
from WANDB import config
from VV_train import load_fsdd, train

wandb.init(project="my-test-project", entity="pilviosol", name='OVERFIT_XX', config=config)
# set_gpu(-1)


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/FW_normalised_flute/'
path_features_matching_vocal_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/FW_normalised_vocal/'


x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_train)


LEARNING_RATE = config['OVERFIT_learning_rate']
BATCH_SIZE = config['OVERFIT_batch_size']
EPOCHS = config['OVERFIT_epochs']


def train_overfit(x_train, y_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(512, 256, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=config['latent_dim']
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train_overfit(x_train, y_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    print('ollare')
    x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    y_train = load_fsdd(y_train_SPECTROGRAMS_PATH)
    autoencoder = train_overfit(x_train, y_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("/nas/home/spol/Thesis/saved_model/OVERFIT")