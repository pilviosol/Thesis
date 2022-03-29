from VV_autoencoder import VAE
from functions import feature_calculation, denormalise_given_min_max, min_max_array_saving, fw_normalise
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_flute_TEST/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/features_matching_flute_TEST/"
test_path_file_vocal = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_vocal_TEST/"
test_path_feature_vocal = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/features_matching_vocal_TEST/"

normalised_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/normalised_flute_features_TEST/"
generated_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/generated_vocal_features_TEST/"
min_max_flute_inference_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/flute_min_max_TEST.npy"
min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/flute_"
denormalised_spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/denormalised_spectrogram_TEST/"
generated_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/generated_vocal_TEST/"

print('PATH, VARIABLES..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
feature_calculation(test_path_file, test_path_feature)
feature_calculation(test_path_file_vocal, test_path_feature_vocal)

print('CALCULATE SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(test_path_feature)
flutes_in_basepath = files_dir.iterdir()
files_dir = pathlib.Path(test_path_feature_vocal)
vocals_in_basepath = files_dir.iterdir()

for flute, vocal in zip(sorted(flutes_in_basepath), sorted(vocals_in_basepath)):
    flute_name = flute.name
    flute_spectrogram = np.load(flute)
    flute_spectrogram = flute_spectrogram[0:512, 0:256]
    vocal_name = vocal.name
    vocal_spectrogram = np.load(vocal)
    vocal_spectrogram = vocal_spectrogram[0:512, 0:256]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img1 = ax1.imshow(flute_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax1.set_title(flute_name[0:-20])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    img2 = ax2.imshow(vocal_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax2.set_title(vocal_name[0:-20])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)

    plt.show()
    plt.close()

print('PLOT THE SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
min_max_flute = min_max_array_saving(test_path_feature, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(test_path_feature, normalised_test_path_feature, min_max_flute)

print('NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX..........ok')

'''
# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE NORMALIZED SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(normalised_test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    spectrogram = np.load(file)
    spectrogram = spectrogram[0:512, 0:256]

    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title('NORMALIZED' + name)
    plt.colorbar()
    # plt.show()
    plt.close()

print('PLOT THE NORMALIZED SPECTROGRAMS..........ok')
'''
# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
vae = VAE.load("/nas/home/spol/Thesis/saved_model/VV_model_FW_x_train_y_train_x_val_y_val")

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
# PLOT THE INPUT AND OUTPUT OF THE NETWORK
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(normalised_test_path_feature)
input_in_basepath = files_dir.iterdir()
files_dir = pathlib.Path(generated_test_path_feature)
output_in_basepath = files_dir.iterdir()

for input, output in zip(sorted(input_in_basepath), sorted(output_in_basepath)):
    input_name = input.name
    input_spectrogram = np.load(input)
    input_spectrogram = input_spectrogram[0:512, 0:256]
    output_name = output.name
    output_spectrogram = np.load(output)
    output_spectrogram = output_spectrogram[0:512, 0:256]
    out_spectrogram = np.squeeze(output_spectrogram)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img1 = ax1.imshow(input_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax1.set_title('INPUT' + input_name[30:-20])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    img2 = ax2.imshow(out_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax2.set_title('OUTPUT' + output_name[30:-20])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)

    plt.show()
    plt.close()

print('PLOT THE OUTPUT OF THE NETWORK..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE THEM
# ---------------------------------------------------------------------------------------------------------------------

# min_max_values = np.load(min_max_flute_inference_path_file)
min_max_values = flute_folder_min_max

files_dir = pathlib.Path(generated_test_path_feature)
files_in_basepath = files_dir.iterdir()
for idx, file in enumerate(files_in_basepath):
    name = file.name
    gen_spectrogram = np.load(file)
    gen_spectrogram = np.squeeze(gen_spectrogram)
    denormalised_spectrogram = denormalise_given_min_max(gen_spectrogram, min_max_values[0][0], min_max_values[0][1])
    # denormalised_spectrogram = denormalise(gen_spectrogram, min_max_values[0][0], min_max_values[0][1])
    np.save(denormalised_spectrogram_path + name, denormalised_spectrogram)

print('GET THE OUTPUT AND DE-NORMALISE THEM..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in sorted(files_in_basepath):
    name = file.name
    print(name)
    denorm_spectrogram = np.load(file)
    denorm_spectrogram = 10 ** (denorm_spectrogram / 10) - 1e-1
    fig = plt.figure()
    img = plt.imshow(denorm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title(name)
    plt.colorbar()
    plt.show()
    plt.close()

print('PLOT THE RECONSTRUCTED SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# SYNTHESIZE THE NEW AUDIO
# ---------------------------------------------------------------------------------------------------------------------

files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in sorted(files_in_basepath):
    name = file.name
    name = name[0:-4]
    print(name)
    denorm_spectrogram = np.load(file)
    reconstructed = librosa.griffinlim(denorm_spectrogram)
    scipy.io.wavfile.write(generated_vocal_path + 'REVERSED_' + name + '.wav', 16000, reconstructed)

print('debug')
