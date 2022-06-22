import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
import pathlib
from functions import load_fsdd, interpolate, denormalise, parabolic_interpolate, spheric_interpolate
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
from WANDB import config
import os
import shutil

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------

main_path = "/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/"
inputs_path = main_path + "inputs/"
inputs_iterdir = pathlib.Path(inputs_path).iterdir()
inputs_new_name_path = main_path + "inputs_new_names/"
inputs_new_name_iterdir = pathlib.Path(inputs_new_name_path).iterdir()

inputs = []
"""
for idx, file in enumerate(sorted(inputs_iterdir)):
    index = str(idx)
    os.rename(inputs_path + file.name, inputs_new_name_path + index + ".npy") """

KS_path = main_path + "KS/"
KG_path = main_path + "KG/"
GO_path = main_path + "GO/"
SO_path = main_path + "SO/"

n_points = 5


# ---------------------------------------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------------------------------------


vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/09-06-2022_01:01")


# ---------------------------------------------------------------------------------------------------------------------
# PERFORM AND SAVE INTERPOLATIONS
# ---------------------------------------------------------------------------------------------------------------------
"""
for inp in sorted(inputs_new_name_iterdir):
    file = np.load(inp)
    print(file.shape)
    inputs.append(file) """

spectrograms = load_fsdd(inputs_new_name_path)

zeros = np.zeros([512, 256], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))

ones = np.ones([512, 256], dtype=float)
ones = np.expand_dims(ones, (-1, 0))

twos = np.add(ones, ones)
threes = np.add(ones, twos)

cond_enc = np.concatenate((ones, zeros, twos, threes), axis=0)

cond0001 = np.asarray([0, 0, 0, 1])
cond0001 = np.expand_dims(cond0001, axis=0)

cond0010 = np.asarray([0, 0, 1, 0])
cond0010 = np.expand_dims(cond0010, axis=0)

cond0100 = np.asarray([0, 1, 0, 0])
cond0100 = np.expand_dims(cond0100, axis=0)

cond1000 = np.asarray([1, 0, 0, 0])
cond1000 = np.expand_dims(cond1000, axis=0)

cond_dec = np.concatenate((cond0001, cond0010, cond0100, cond1000), axis=0)


for idx, spectrogram in enumerate(spectrograms):
    spectrogram = np.expand_dims(spectrogram, axis=0)
    index = str(idx)
    concat = []
    concat = np.concatenate((spectrogram, spectrogram, spectrogram, spectrogram), axis=0)

    spectrogram_full = [concat, cond_enc, cond_dec]

    encoded_spectrograms = vae.encoder.predict(spectrogram_full)

    row0 = np.asarray(interpolate(encoded_spectrograms[1].flatten(), encoded_spectrograms[0].flatten(), n=n_points))
    generated_row0 = vae.decoder.predict(row0)
    for i, element in enumerate(generated_row0):
        i = str(i)
        np.save(KS_path + index + "_KS_" + i, element)

    row4 = np.asarray(interpolate(encoded_spectrograms[2].flatten(), encoded_spectrograms[3].flatten(), n=n_points))
    generated_row4 = vae.decoder.predict(row4)
    for i, element in enumerate(generated_row4):
        i = str(i)
        np.save(GO_path + index + "_GO_" + i, element)

    column0 = np.asarray(interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[3].flatten(), n=n_points))
    generated_column0 = vae.decoder.predict(column0)
    for i, element in enumerate(generated_column0):
        i = str(i)
        np.save(SO_path + index + "_SO_" + i, element)

    column4 = np.asarray(interpolate(encoded_spectrograms[1].flatten(), encoded_spectrograms[2].flatten(), n=n_points))
    generated_column4 = vae.decoder.predict(column4)
    for i, element in enumerate(generated_column4):
        i = str(i)
        np.save(KG_path + index + "_KG_" + i, element)

print('debug')











