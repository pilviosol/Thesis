import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
import pathlib
from functions import load_fsdd, interpolate, denormalise
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
from WANDB import config
import os

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------


folder_number = str(9)
spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237/inputs/input" + folder_number + "/"
spectrogram = pathlib.Path(spectrogram_path)
images_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237/images/image" + folder_number + "/"
wav_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237/wavs/wav" + folder_number + "/"
tsne_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237/TSNEs/"
min_max = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/string_folder_min_max.npy"

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
generated_image_vectors01 = np.asarray(
    interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[1].flatten(), n=4))
generated_image_vectors12 = np.asarray(
    interpolate(encoded_spectrograms[1].flatten(), encoded_spectrograms[2].flatten(), n=4))
generated_image_vectors23 = np.asarray(
    interpolate(encoded_spectrograms[2].flatten(), encoded_spectrograms[3].flatten(), n=4))
generated_image_vectors30 = np.asarray(
    interpolate(encoded_spectrograms[3].flatten(), encoded_spectrograms[0].flatten(), n=4))

generated_spectrograms01 = vae.decoder.predict(generated_image_vectors01)
generated_spectrograms12 = vae.decoder.predict(generated_image_vectors12)
generated_spectrograms23 = vae.decoder.predict(generated_image_vectors23)
generated_spectrograms30 = vae.decoder.predict(generated_image_vectors30)

# ---------------------------------------------------------------------------------------------------------------------
# TSNE ON GENERATED POINTS
# ---------------------------------------------------------------------------------------------------------------------

"""
encoded_generated01 = vae.tsne_interpolation(generated_image_vectors01, perplexity=3, title='x_val01',
                                             annotations=annotations, color='red',
                                             save_image_path=tsne_path + folder_number + '_01')
encoded_generated12 = vae.tsne_interpolation(generated_image_vectors12, perplexity=3, title='x_va12l',
                                             annotations=annotations, color='green',
                                             save_image_path=tsne_path + folder_number + '_12')
encoded_generated23 = vae.tsne_interpolation(generated_image_vectors23, perplexity=3, title='x_val23',
                                             annotations=annotations, color='blue',
                                             save_image_path=tsne_path + folder_number + '_23')
encoded_generated30 = vae.tsne_interpolation(generated_image_vectors30, perplexity=3, title='x_val30',
                                             annotations=annotations, color='black',
                                             save_image_path=tsne_path + folder_number + '_30')
"""
# ---------------------------------------------------------------------------------------------------------------------
# PLOT PREDICTIONS, SAVE IMAGES, RE-SYNTHESIZE AND SAVE AUDIO
# ---------------------------------------------------------------------------------------------------------------------


min_and_max = np.load(min_max)
minimum = min_and_max[0][1]
maximum = min_and_max[0][0]


def plot_save_interpolations(generated_spectrograms, couple):
    try:
        os.mkdir(images_path + couple + '/')
        os.mkdir(wav_path + couple + '/')
    except OSError:
        print("Creation of the directory  failed")

    for idx, file in enumerate(generated_spectrograms):
        spectrogram = np.squeeze(file)

        """
        fig = plt.figure()
        img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
        # plt.colorbar()
        plt.tight_layout()
        plt.title(idx)
        plt.savefig(images_path + couple + '/' + str(idx))
        plt.show()
        plt.close()
        """

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
        fig.savefig(images_path + couple + '/' + str(idx))
        plt.show()
        plt.close()


        denormalised_spectrogram = denormalise(spectrogram, minimum, maximum)
        spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
        reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
        scipy.io.wavfile.write(wav_path + couple + '/' + str(idx) + '.wav', SR, reconstructed)


plot_save_interpolations(generated_spectrograms01, '01')
plot_save_interpolations(generated_spectrograms12, '12')
plot_save_interpolations(generated_spectrograms23, '23')
plot_save_interpolations(generated_spectrograms30, '30')

print('debug')
