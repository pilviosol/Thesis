from VV_autoencoder import VAE
from functions import feature_calculation, normalise_set_and_save_min_max
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import pickle

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute_features/"
normalised_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/normalised_testing_flute_features/"
min_max_flute_inference_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/inference_flute_min_max.npy"
denormalised_spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/denormalised_spectrogram/"

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
    spectrogram = np.load(file)
    img = librosa.display.specshow(spectrogram, y_axis='linear', x_axis='time', ax=ax)
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
    spectrogram = np.load(file)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print('spectrogram expanded dims: ', spectrogram.shape)

    generated_spectrogram, latent_representation = generate(spectrogram)


# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE IT
# ---------------------------------------------------------------------------------------------------------------------

min_max_values = np.load(min_max_flute_inference_path_file)
generated_spectrogram = np.squeeze(generated_spectrogram)


def denormalise(norm_array, original_min, original_max):
    array = (norm_array - norm_array.min()) / (norm_array.max() - norm_array.min())
    array = array * (original_max - original_min) + original_min
    return array


denormalised_spectrogram = denormalise(generated_spectrogram, min_max_values[0][0], min_max_values[0][1])
np.save(denormalised_spectrogram_path + 'denormalised_spectrogram.npy', denormalised_spectrogram)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    print(file)
    fig, ax = plt.subplots()
    spectrogram = np.load(file)
    img = librosa.display.specshow(spectrogram, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Denormalised spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# SYNTHESIZE THE NEW AUDIO
# ---------------------------------------------------------------------------------------------------------------------




print('debug')