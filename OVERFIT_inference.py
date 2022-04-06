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

x_train_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/FW_normalised_flute/"
y_train_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/FW_normalised_vocal/"
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
    print('generated_spectrogram.shape:', generated_spectrogram.shape)
    np.save(generated_path + 'GENERATED_' + name, generated_spectrogram)


VAE_input = pathlib.Path(x_train_path).iterdir()
VAE_output = pathlib.Path(generated_path).iterdir()
VAE_expected_output = pathlib.Path(y_train_path).iterdir()


for inpt, output, expected_output in zip(sorted(VAE_input), sorted(VAE_output), sorted(VAE_expected_output)):
    input_name = inpt.name
    print('input_name: ', input_name)
    input_spectrogram = np.load(inpt)
    input_spectrogram = input_spectrogram[0:512, 0:256]

    output_name = output.name
    print('output_name: ', output_name)
    output_spectrogram = np.load(output)
    out_spectrogram = np.squeeze(output_spectrogram)
    output_spectrogram = output_spectrogram[0:512, 0:256]

    expected_output_name = expected_output.name
    expected_output_spectrogram = np.load(expected_output)
    expected_output_spectrogram = expected_output_spectrogram[0:512, 0:256]

    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3)

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

    # EXPECTED OUTPUT
    img5 = ax5.imshow(expected_output_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax5.set_title('EXPECTED OUTPUT_' + output_name[44:-16])
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    cbar5 = plt.colorbar(img4, cax=cax4)

    plt.show()
    plt.close()

    print(np.mean((out_spectrogram-expected_output_spectrogram)**2))


