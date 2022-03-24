from VV_autoencoder import VAE
from functions import feature_calculation, normalise_set_and_save_min_max, denormalise
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
import pickle

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_flute_TEST/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/features_matching_flute_TEST/"
normalised_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/normalised_flute_features_TEST/"
generated_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/generated_vocal_features_TEST/"
min_max_flute_inference_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/flute_min_max_TEST.npy"
denormalised_spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/denormalised_spectrogram_TEST/"
generated_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/generated_vocal_TEST/"


print('PATH, VARIABLES..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
feature_calculation(test_path_file, test_path_feature)


print('CALCULATE SPECTROGRAMS..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    spectrogram = np.load(file)
    spectrogram = spectrogram[0:512, 0:256]
    '''
    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title(name)
    plt.colorbar()
    plt.show() '''


print('PLOT THE SPECTROGRAMS..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
min_max_flute_inference = normalise_set_and_save_min_max(test_path_feature, normalised_test_path_feature)
np.save(min_max_flute_inference_path_file, min_max_flute_inference)


print('NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
vae = VAE.load("/nas/home/spol/Thesis/saved_model/VV_model_x_train_x_train_x_val_x_val")


print('IMPORT THE MODEL..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# FEED THEM TO THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


files_dir = pathlib.Path(normalised_test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    spectrogram = np.load(file)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print('spectrogram expanded dims: ', spectrogram.shape)

    generated_spectrogram, latent_representation = generate(spectrogram)
    np.save(generated_test_path_feature + 'GENERATED_' + name, generated_spectrogram)


print('FEED THEM TO THE MODEL..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE THEM
# ---------------------------------------------------------------------------------------------------------------------

min_max_values = np.load(min_max_flute_inference_path_file)

files_dir = pathlib.Path(generated_test_path_feature)
files_in_basepath = files_dir.iterdir()
for idx, file in enumerate(files_in_basepath):
    name = file.name
    gen_spectrogram = np.load(file)
    gen_spectrogram = np.squeeze(gen_spectrogram)
    denormalised_spectrogram = denormalise(gen_spectrogram, min_max_values[idx][0], min_max_values[idx][1])
    np.save(denormalised_spectrogram_path + name, denormalised_spectrogram)


print('GET THE OUTPUT AND DE-NORMALISE THEM..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    print(name)
    denorm_spectrogram = np.load(file)
    denorm_spectrogram = 10**(denorm_spectrogram/10) - 1e-1
    fig = plt.figure()
    img = plt.imshow(denorm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title('ollare')
    plt.colorbar()
    plt.show()


print('PLOT THE RECONSTRUCTED SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# SYNTHESIZE THE NEW AUDIO
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    name = name[0:-4]
    print(name)
    denorm_spectrogram = np.load(file)
    reconstructed = librosa.griffinlim(denorm_spectrogram)
    scipy.io.wavfile.write(generated_vocal_path + 'REVERSED_' + name + '.wav', 16000, reconstructed)


print('debug')

