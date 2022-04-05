from VV_autoencoder import VAE
from functions import feature_calculation, denormalise, normalise_set_and_save_min_max
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from WANDB import config
import os

x_train_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/normalised_flute_ls/"
generated_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/generated_vocal/"
# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
vae = VAE.load("/nas/home/spol/Thesis/saved_model/OVERFIT")


def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


normalised_features_path = pathlib.Path(x_train_path).iterdir()

for file in sorted(normalised_features_path):
    name = file.name
    spectrogram = np.load(file)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print('spectrogram expanded dims: ', spectrogram.shape)

    generated_spectrogram, latent_representation = generate(spectrogram)
    np.save(generated_path + 'GENERATED_' + name, generated_spectrogram)


VAE_input = pathlib.Path(x_train_path).iterdir()
VAE_output = pathlib.Path(generated_path).iterdir()

for inpt, output in zip(sorted(VAE_input), sorted(VAE_output)):
    input_name = inpt.name
    print('input_name: ', input_name)
    input_spectrogram = np.load(inpt)
    input_spectrogram = input_spectrogram[0:512, 0:256]
    output_name = output.name
    print('output_name: ', output_name)
    output_spectrogram = np.load(output)
    output_spectrogram = output_spectrogram[0:512, 0:256]
    out_spectrogram = np.squeeze(output_spectrogram)

    fig2, (ax3, ax4) = plt.subplots(1, 2)

    # VAE INPUT
    img3 = ax3.imshow(input_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax3.set_title('INPUT_' + input_name[34:-16])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(img3, cax=cax3)

    # VAE OUTPUT
    img4 = ax4.imshow(out_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax4.set_title('OUTPUT_' + output_name[44:-16])
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(img4, cax=cax4)

    plt.show()
    plt.close()

    print(np.mean((out_spectrogram-input_spectrogram)**2))


