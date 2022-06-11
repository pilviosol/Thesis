from CVAE_autoencoder_multi import CVAEMulti
from functions import denormalise
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
from WANDB import config
from functions import load_fsdd


# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES, DATE
# ---------------------------------------------------------------------------------------------------------------------


with open('/nas/home/spol/Thesis/last_date.txt') as f:
    date = f.read()
    print('date: ', date)

normalised_flute_features_TEST = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/NORMALIZED_flute/"
path_save_figures = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/IMAGES/IMAGES_0906_0101_NAMES_ok/"
generated_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/GENERATED/GENERATED_0906_0101_NAMES_ok/"
SR = config['sample_rate']


# ---------------------------------------------------------------------------------------------------------------------
# CREATE CONDITIONING LABELS
# ---------------------------------------------------------------------------------------------------------------------


# ENCODER CONDITIONING MATRICES
zeros = np.zeros([512, 256], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros_test = np.repeat(zeros, 88, axis=0)

ones = np.ones([512, 256], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones_test = np.repeat(ones, 88, axis=0)

twos_test = np.add(ones_test, ones_test)

threes_test = np.add(twos_test, ones_test)

cond_enc_test = np.concatenate((ones_test, zeros_test, twos_test, threes_test), axis=0)


# DECODER CONDITIONING VECTORS
cond0001 = np.asarray([0, 0, 0, 1])
cond0001 = np.expand_dims(cond0001, axis=0)
cond0001_test = np.repeat(cond0001, 88, axis=0)

cond0010 = np.asarray([0, 0, 1, 0])
cond0010 = np.expand_dims(cond0010, axis=0)
cond0010_test = np.repeat(cond0010, 88, axis=0)

cond0100 = np.asarray([0, 1, 0, 0])
cond0100 = np.expand_dims(cond0100, axis=0)
cond0100_test = np.repeat(cond0100, 88, axis=0)

cond1000 = np.asarray([1, 0, 0, 0])
cond1000 = np.expand_dims(cond1000, axis=0)
cond1000_test = np.repeat(cond1000, 88, axis=0)

cond_dec_test = np.concatenate((cond0001_test, cond0010_test, cond0100_test, cond1000_test), axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL AND DEFINE GENERATE FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/09-06-2022_01:01")

def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


# ---------------------------------------------------------------------------------------------------------------------
# LOAD NORMALIZED TEST SPECTROGRAMS AND ORGANIZE THEM
# ---------------------------------------------------------------------------------------------------------------------
normalised_features_path = pathlib.Path(normalised_flute_features_TEST)
x_test0 = load_fsdd(normalised_features_path)

names_string = []
names_keyboard = []
names_guitar = []
names_organ = []

for element in sorted(normalised_features_path.iterdir()):
    name = element.name
    names_string.append("STRING_" + name[0:-4])
    names_keyboard.append("KEYBOARD_" + name[0:-4])
    names_guitar.append("GUITAR_" + name[0:-4])
    names_organ.append("ORGAN_" + name[0:-4])

names = np.concatenate((names_string, names_keyboard, names_guitar, names_organ), axis=0)

x_test = np.concatenate((x_test0, x_test0, x_test0, x_test0), axis=0)
x_test = [x_test, cond_enc_test, cond_dec_test]

spectrograms, latent_representations = vae.reconstruct(x_test)

for idx, element in enumerate(spectrograms):
    element = np.squeeze(element)
    print(element.shape)
    print(idx)

    fig = plt.figure()
    img = plt.imshow(element, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title(str(idx))
    plt.colorbar()
    # plt.savefig(path_save_figures + str(idx))
    # plt.show()
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
    img = plt.imshow(denormalised_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title(names[idx])
    plt.colorbar()
    plt.savefig(path_save_figures + names[idx])
    # plt.show()
    plt.close()

    spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
    scipy.io.wavfile.write(generated_path + names[idx] + '.wav', SR, reconstructed)


print('debaggone')