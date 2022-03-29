import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pathlib
from functions import normalise_set_and_save_min_max, feature_calculation, denormalise


test_normalization_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/TEST_VARI/testing_normalization/"
test_normalization_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/TEST_VARI/testing_normalization_features/"
normalised_test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET" \
                               "/TEST_VARI/normalised_testing_normalization_features/"
denormalised_spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET" \
                                "/TEST_VARI/denormalised_spectrogram_test_normalization/"


# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE THE SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
feature_calculation(test_normalization_path_file, test_normalization_path_feature)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(test_normalization_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    fig, ax = plt.subplots()
    INIZIALE_spectrogram = np.load(file)
    INIZIALE_spectrogram = INIZIALE_spectrogram[0:512, 0:256]
    img = librosa.display.specshow(INIZIALE_spectrogram, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Test Sample Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAM AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
min_max_test_inference = normalise_set_and_save_min_max(test_normalization_path_feature, normalised_test_path_feature)
# np.save(min_max_flute_inference_path_file, min_max_flute_inference)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE NORMALISED SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(normalised_test_path_feature)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    fig, ax = plt.subplots()
    NORMALIZZATO_spectrogram = np.load(file)
    img = librosa.display.specshow(NORMALIZZATO_spectrogram, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('NORMALISED Test Sample Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE IT
# ---------------------------------------------------------------------------------------------------------------------
normalised_spectrogram = np.load('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET'
                                 '/normalised_testing_normalization_features/normalised_021_vocal_synthetic_005-021'
                                 '-025_STFTMAG.npy')
denormalised_spectrogram = denormalise(normalised_spectrogram, min_max_test_inference[0][0], min_max_test_inference[0][1])
np.save(denormalised_spectrogram_path + 'denormalised_spectrogram.npy', denormalised_spectrogram)

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE DENORMALISED SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(denormalised_spectrogram_path)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    fig, ax = plt.subplots()
    DENORMALIZZATO_spectrogram = np.load(file)
    img = librosa.display.specshow(DENORMALIZZATO_spectrogram, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('DENORMALISED Test Sample Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


print('debug')
