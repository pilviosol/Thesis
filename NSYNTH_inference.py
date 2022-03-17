from VV_autoencoder import VAE
from functions import feature_calculation, normalise_set_and_save_min_max
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute_features/"
normalised_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/normalised_testing_flute_features/"
min_max_flute_inference_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/inference_flute_min_max.npy"


# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE THE SPECTROGRAM SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
feature_calculation(test_path_file, test_path_feature)

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:

    fig, ax = plt.subplots()
    img = librosa.display.specshow(file, y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Flute Sample Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAM AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
min_max_flute_inference = normalise_set_and_save_min_max(test_path_feature, normalised_test_path_feature)
np.save(min_max_flute_inference_path_file, min_max_flute_inference)
# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
vae = VAE.load("/nas/home/spol/Thesis/saved_model/VV_model")


# ---------------------------------------------------------------------------------------------------------------------
# FEED IT TO THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


files_dir = pathlib.Path(normalised_test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    generated_spectrogram, latent_representation = generate(file)


# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE IT
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# SYNTHESIZE THE NEW AUDIO
# ---------------------------------------------------------------------------------------------------------------------




