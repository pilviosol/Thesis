import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
import pathlib
from functions import load_fsdd, interpolate, denormalise
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
from WANDB import config
import os

"""

RIFARE FACENDO BENE LE CARTELLE E SALVANDO DECENTEMENTE LE COSE IN MANIERA AUTOMATIZZATA

"""

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
main_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/"
date = "03062022/"
folder_number = str(9)
n_points = 5

try:
    os.mkdir(main_path + date)
except OSError:
    print("Creation of the directory  failed")

spectrogram_path = main_path + "inputs/input" + folder_number + "/"
spectrogram = pathlib.Path(spectrogram_path)

images_path = main_path + date + "images/"
wavs_path = main_path + date + "wavs/"
save_interpolation_path = main_path + date + "INTERPOLATIONs/"

try:
    os.mkdir(images_path)
    os.mkdir(wavs_path)
    os.mkdir(save_interpolation_path)
except OSError:
    print("Creation of the directory  failed")

min_max = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/string_folder_min_max.npy"
min_and_max = np.load(min_max)
minimum = min_and_max[0][1]
maximum = min_and_max[0][0]

SR = config['sample_rate']
annotations = []
for i in range(10):
    annotations.append(str(i))

# ---------------------------------------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------------------------------------


vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/18-05-2022_22:37")

# ---------------------------------------------------------------------------------------------------------------------
# FEED THE MODEL, INTERPOLATE AND PREDICT INTERPOLATION_multi
# ---------------------------------------------------------------------------------------------------------------------

spectrogram = load_fsdd(spectrogram)
spectrograms = np.concatenate((spectrogram, spectrogram, spectrogram, spectrogram), axis=0)

zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))

ones = np.ones([512, 64], dtype=float)
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

spectrograms_full = [spectrograms, cond_enc, cond_dec]

encoded_spectrograms = vae.encoder.predict(spectrograms_full)

# ---------------------------------------------------------------------------------------------------------------------
# COLUMN
# ---------------------------------------------------------------------------------------------------------------------

# BOUNDARIES (UPPER AND LOWER)
generated_image_vectors01 = np.asarray(
    interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[1].flatten(), n=n_points))
generated_image_vectors23 = np.asarray(
    interpolate(encoded_spectrograms[2].flatten(), encoded_spectrograms[3].flatten(), n=n_points))

# COLUMNS
for i in range(n_points):
    print(i)

    path_interpolations = save_interpolation_path + '/column' + str(i)
    path_images = images_path + '/column' + str(i)
    path_wavs = wavs_path + '/column' + str(i)
    try:
        os.mkdir(path_interpolations)
        os.mkdir(path_images)
        os.mkdir(path_wavs)
    except OSError:
        print("Creation of the directory  failed")

    generated_col = np.asarray(interpolate(generated_image_vectors01[i].flatten(),
                                           generated_image_vectors23[n_points - 1 - i].flatten(), n=n_points))
    generated_spectrograms = vae.decoder.predict(generated_col)
    for idx, element in enumerate(generated_spectrograms):
        np.save(path_interpolations + "/" + str(idx), element)

        spectrogram = np.squeeze(element)
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
        fig.savefig(path_images + "/" + str(idx))
        plt.close()

        denormalised_spectrogram = denormalise(spectrogram, minimum, maximum)
        spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
        reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
        scipy.io.wavfile.write(path_wavs + '/' + str(idx) + '.wav', SR, reconstructed)

# ---------------------------------------------------------------------------------------------------------------------
# ROWS
# ---------------------------------------------------------------------------------------------------------------------
# BOUNDARIES (LEFT AND RIGHT)

generated_image_vectors12 = np.asarray(
    interpolate(encoded_spectrograms[1].flatten(), encoded_spectrograms[2].flatten(), n=n_points))
generated_image_vectors30 = np.asarray(
    interpolate(encoded_spectrograms[3].flatten(), encoded_spectrograms[0].flatten(), n=n_points))

# ROWS
for i in range(n_points):
    print(i)

    path_interpolations = save_interpolation_path + '/row' + str(i)
    path_images = images_path + '/row' + str(i)
    path_wavs = wavs_path + '/row' + str(i)
    try:
        os.mkdir(path_interpolations)
        os.mkdir(path_images)
        os.mkdir(path_wavs)
    except OSError:
        print("Creation of the directory  failed")

    generated_row = np.asarray(interpolate(generated_image_vectors12[i].flatten(),
                                           generated_image_vectors30[n_points - 1 - i].flatten(), n=n_points))
    generated_spectrograms = vae.decoder.predict(generated_row)
    for idx, element in enumerate(generated_spectrograms):
        np.save(path_interpolations + "/" + str(idx), element)

        spectrogram = np.squeeze(element)
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
        fig.savefig(path_images + "/" + str(idx))
        plt.close()

        denormalised_spectrogram = denormalise(spectrogram, minimum, maximum)
        spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
        reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
        scipy.io.wavfile.write(path_wavs + '/' + str(idx) + '.wav', SR, reconstructed)


print('debug')
