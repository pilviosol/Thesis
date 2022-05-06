from VV_autoencoder import VAE
from functions import feature_calculation, denormalise, fw_normalise, min_max_array_saving
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from WANDB import config
import os
from functions import load_fsdd
# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
with open('/nas/home/spol/Thesis/last_date.txt') as f:
    date = f.read()
    print('date: ', date)

normalised_flute_features_TEST = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/FW_normalised_flute_0605_TEST/"
path_save_figures = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/IMAGES_0605/"
SR =16000
generated_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/GENERATED/GENERATED_0605/"

# ---------------------------------------------------------------------------------------------------------------------
# CREATE CONDITIONING LABELS
# ---------------------------------------------------------------------------------------------------------------------


# ENCODER CONDITIONING MATRICES
ones = np.ones([512, 64], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones_test = np.repeat(ones, 14, axis=0)

zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros_test = np.repeat(zeros, 14, axis=0)

cond_enc_test = np.concatenate((ones_test, zeros_test), axis=0)


# DECODER CONDITIONING VECTORS
cond01 = np.asarray([0, 1])
cond01 = np.expand_dims(cond01, axis=0)
cond01_test = np.repeat(cond01, 14, axis=0)

cond10 = np.asarray([1, 0])
cond10 = np.expand_dims(cond10, axis=0)
cond10_test = np.repeat(cond10, 14, axis=0)

cond_dec_test = np.concatenate((cond01_test, cond10_test), axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL AND DEFINE GENERATE FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = VAE.load("/nas/home/spol/Thesis/saved_model/CVAE/06-05-2022_01:36")

def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


# ---------------------------------------------------------------------------------------------------------------------
# LOAD NORMALIZED TEST SPECTROGRAMS AND ORGANIZE THEM
# ---------------------------------------------------------------------------------------------------------------------
normalised_features_path = pathlib.Path(normalised_flute_features_TEST)

x_test0 = load_fsdd(normalised_features_path)
x_test1 = load_fsdd(normalised_features_path)
x_test = np.concatenate((x_test0, x_test1), axis=0)
x_test = [x_test, cond_enc_test, cond_dec_test]

spectrograms, latent_representations = vae.reconstruct(x_test)

for idx, element in enumerate(spectrograms):
    element = np.squeeze(element)
    print(element.shape)
    print(idx)

    fig = plt.figure()
    img = plt.imshow(element, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
    plt.title(str(idx))
    plt.colorbar()
    plt.savefig(path_save_figures + str(idx))
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------------------------------------------------
# DE-NORMALISE OUTPUTS
# ---------------------------------------------------------------------------------------------------------------------

min_max_keyboard = np.load('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/keyboard_folder_min_max.npy')

for idx, element in enumerate(spectrograms):
    element = np.squeeze(element)
    print(element.shape)
    print(idx)
    denormalised_spectrogram = denormalise(element, min_max_keyboard[0][1], min_max_keyboard[0][0])

    fig = plt.figure()
    img = plt.imshow(denormalised_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
    plt.title("denormalised_" + str(idx))
    plt.colorbar()
    plt.savefig(path_save_figures + "denormalised_" + str(idx))
    plt.show()
    plt.close()

    spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
    scipy.io.wavfile.write(generated_path + "denormalised_" + str(idx) + '.wav', SR, reconstructed)


print('debaggone')