import os
import numpy as np
import pathlib
import shutil
import math
import librosa
import librosa.display
import scipy.io.wavfile
from WANDB import config

# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
CONSTANT_256 = 256
N_FFT = config['n_fft']
HOP_LENGTH = config['hop_length']
WIN_LENGTH = config['win_length']


def extract_subset(origin_path, destination_path, string):
    files_dir = pathlib.Path(origin_path)
    files_in_basepath = files_dir.iterdir()
    try:
        shutil.rmtree((destination_path + string))
    except OSError:
        print("Removal of the directory %s failed" % (destination_path + string))
    else:
        print("Successfully removed the directory %s " % (destination_path + string))

    os.mkdir(destination_path + string)

    for item in files_in_basepath:
        name = item.name
        if string in str(name):
            print('Moving: ', name)
            print('--------------------------------------------')
            shutil.copyfile(origin_path + name, destination_path + string + '/' + name)


def divider_256(origin_path, destination_path):
    """

    :param origin_path: path where the features are extracted
    :param destination_path: path where to store the features divided in 256 long chunks
    :return: void

    """
    features_dir = pathlib.Path(origin_path)
    features_in_basepath = features_dir.iterdir()

    for item in features_in_basepath:
        name = item.name[0:-4]
        print(name)
        feature = np.load(item)
        length = feature.shape[1]
        chunks = math.floor(length / CONSTANT_256)
        print(chunks)
        for i in range(chunks):
            chunk = feature[0:512, i * CONSTANT_256: (i + 1) * CONSTANT_256]
            print("chunk.shape: ", chunk.shape)
            np.save(destination_path + name + "_chunk_" + str(i), chunk)
            print("i: ", i)
        print('--------------------------------------------')


def extract_features(file_name):
    """

    :param file_name: file to be analyzed
    :return:
    cqt: constnt Q transform
    stft_mag: magnitude of STFT
    stft_mag_real: real part of stft magnitude
    stft_mag_imag: imaginary part of stft magnitude
    stft_phase: phase of stft
    mel_spectrogram: spectrogram using mel coefficients

    """
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', mono=True, sr=None)
        print('sample_rate: ', sample_rate)
        # cqt = librosa.cqt(y=audio, sr=sample_rate, hop_length=512, fmin=32.7, filter_scale=0.8,
        # bins_per_octave=48, n_bins=384)
        # stft_full = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        # stft_mag = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=512))
        # stft_mag_real = stft_mag.real
        # stft_mag_imag = stft_mag.imag
        # stft_phase = np.angle(librosa.stft(y=audio, hop_length=512))
        # mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,
        # n_fft=2048, hop_length=512,
        # n_mels=128)
        stft_mag = np.abs(librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
        log_spectrogram = 10 * np.log10(stft_mag + 1e-5)
        log_spectrogram = log_spectrogram[0:512, 0:64]
        # cambiare 1e-5 FARE PROVE

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    # return cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram
    return log_spectrogram


def feature_calculation(path_songs, store_features_path):
    """

    :param path_songs: folder where all songs to be analyzed are
    :param store_features_path: path where to store all the features
    :return: void
    """
    data_dir = pathlib.Path(path_songs)
    files_in_basepath = data_dir.iterdir()

    for item in sorted(files_in_basepath):
        if item.is_file():
            print(item.name)
            name = item.name[0:-4]

            # cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)
            stft_mag = extract_features(item)

            # Saving all features in one folder (.npy format)
            # np.save(store_features_path + name + "_CQT", cqt)
            # np.save(store_features_path + name + "_STFTFULL", stft_full)
            np.save(store_features_path + name + "_STFTMAG_NEW", stft_mag)
            # np.save(store_features_path + name + "_STFTMAG_REAL", stft_mag_real)
            # np.save(store_features_path + name + "_STFTMAG_IMAG", stft_mag_imag)
            # np.save(store_features_path + name + "_STFTPHASE", stft_phase)
            # np.save(store_features_path + name + "_MEL", mel_spectrogram)

        else:
            print('That is not a file')


def resample(origin_path, destination_path, new_sr):
    files_dir = pathlib.Path(origin_path)
    files_in_basepath = files_dir.iterdir()
    for item in files_in_basepath:
        name = item.name
        print(name)
        y, sr = librosa.load(item, sr=None)
        y_new_sr = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
        scipy.io.wavfile.write(destination_path + '/' + 'Resampled_' + name, new_sr, y_new_sr)


def rename_files_by_pitch(path):
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    for item in sorted(files_in_basepath):
        name = item.name
        pitch = str(name)[-11: -8]
        new_name = pitch + '_' + name
        print('new_name: ', new_name)
        os.rename(path + '/' + name, path + '/' + new_name)


def count_pitches(list):
    pitches = []
    for item in sorted(list):
        pitch = str(item)[0:3]
        print('pitch', pitch)
        pitches.append(pitch)
    return sorted(set(pitches))


def append_pitches_velocities(path):
    pitches_velocities = []
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    for item in sorted(files_in_basepath):
        name = item.name
        pitch_velocity = str(name)[-11: -4]
        print('pitch_velocity', pitch_velocity)
        pitches_velocities.append(pitch_velocity)
    return pitches_velocities


def remove_files_if_pitch_not_matching(path, elimination_list):
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    for file in files_in_basepath:
        name = file.name
        if name[-11: -4] in elimination_list:
            os.remove(path + '/' + name)


def how_many_pitches(path, temp_pitch):
    counts = []
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    count = 0
    temp_pitch = temp_pitch
    for file in sorted(files_in_basepath):
        name = file.name
        pitch = name[0:3]
        if pitch == temp_pitch:
            count += 1
            temp_pitch = pitch
        else:
            counts.append(count)
            temp_pitch = pitch
            count = 1
    return counts


def normalise(array):
    """

    :param array: array to be normalizes
    :return:
    norm_array: normalized based on his min and max
    original_min: the min of the array
    original_max: the max of the array

    """
    original_min = array.min()
    original_max = array.max()
    norm_array = (array - array.min()) / (array.max() - array.min())
    return norm_array, original_min, original_max


def normalise_given_min_max(array, minimum, maximum):
    """

    :param array: array to be normalised
    :param minimum: given min val
    :param maximum: given max val
    :return: norm_array: normalized array in range 0-1

    """
    norm_array = (array - minimum) / (maximum - minimum)
    return norm_array


def denormalise_given_min_max(array, minimum, maximum):
    """

    :param array: array to be denormalised
    :param minimum: given min val
    :param maximum: given max val
    :return: norm_array: array in range full range

    """
    array = array * (maximum - minimum) + minimum
    return array


def denormalise(norm_array, original_min, original_max):
    """

    :param norm_array: normalised array to be denormalised
    :param original_min: original min of the array that has been normlised
    :param original_max: original max of the array that has been normalised
    :return: array: the denormalised array

    """
    array = (norm_array - norm_array.min()) / (norm_array.max() - norm_array.min())
    array = array * (original_max - original_min) + original_min
    return array


def min_max_array_saving(spectrograms_path, saving_path):
    """

    :param spectrograms_path: path where all spectrograms are stored
    :param saving_path: path where the min_max_array will be saved
    :return: min_max_array: min max array with all min max values for each spectrogram + saves it at saving_path

    """
    min_max_array = []
    files_dir = pathlib.Path(spectrograms_path)
    files_in_basepath = files_dir.iterdir()
    for file in sorted(files_in_basepath):
        loaded_file = np.load(file)
        loaded_file = loaded_file[0:512, 0:256]
        minimum = loaded_file.min()
        maximum = loaded_file.max()
        min_max_values = np.array([minimum, maximum])
        min_max_array.append(min_max_values)
        np.save(saving_path + 'min_max_array', min_max_array)
    return min_max_array


def normalise_set_and_save_min_max(original_path, new_path):
    """

    :param original_path: path where all spectrograms are stored
    :param new_path: path where all normalised spectrograms will be saved
    :return: min_max_array: min max array with all min max values for each spectrogram \
             and saves all normalised spectrograms (ALL normalised with their own min and max, NOT folder-wise) \
             in new_path

    """
    min_max_array = []
    files_dir = pathlib.Path(original_path)
    files_in_basepath = files_dir.iterdir()
    for file in sorted(files_in_basepath):
        name = file.name
        loaded_file = np.load(file)
        loaded_file = loaded_file[0:512, 0:256]
        normalised_spectrogram, original_min, original_max = normalise(loaded_file)
        min_max_values = np.array([original_min, original_max])
        min_max_array.append(min_max_values)
        np.save(new_path + 'normalised_new_' + name, normalised_spectrogram)
    return min_max_array


def fw_normalise(original_path, new_path, min_max_array):
    """

    :param original_path: path where all spectrograms are stored
    :param new_path: path where all normalised spectrograms will be saved
    :param min_max_array: array with all min_max_values of the spectrograms to be normalised
    :return: void but saves all normalised spectrograms into new_path

    """
    folder_min_max = []
    min_max_np = np.array(min_max_array)
    maximum = min_max_np.max()
    minimum = min_max_np.min()
    folder_min_max.append([maximum, minimum])

    files_dir = pathlib.Path(original_path)
    files_in_basepath = files_dir.iterdir()
    for file in sorted(files_in_basepath):
        name = file.name
        loaded_file = np.load(file)
        loaded_file = loaded_file[0:512, 0:64]
        normalised_loaded_file = normalise_given_min_max(loaded_file, minimum, maximum)
        np.save(new_path + 'normalised_FW_' + name, normalised_loaded_file)
    return folder_min_max


def load_fsdd(spectrograms_path):
    """

    :param spectrograms_path: where the normalised spectrograms are
    :return: x_train, array with all spectrograms data appended

    """
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        count = 0
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            # spectrogram = spectrogram[0:512, 0:64]
            x_train.append(spectrogram)
            count += 1
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (4130, 512, 256, 1)
    return x_train


def load_fsdd_concat(spectrograms_path, label):
    """

    :param spectrograms_path: where the normalised spectrograms are
            label 1 or 0 depending on the label
    :return: x_train, array with all spectrograms data appended

    """
    ones = np.ones([512, 64], dtype=float)
    ones = np.expand_dims(ones, -1)
    zeros = np.zeros([512, 64], dtype=float)
    zeros = np.expand_dims(zeros, -1)

    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        count = 0
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            # spectrogram = spectrogram[0:512, 0:64]
            spectrogram = np.expand_dims(spectrogram, -1)

            if label == 1:
                spectrogram = np.concatenate((spectrogram, ones), axis=2)
            else:
                spectrogram = np.concatenate((spectrogram, ones), axis=2)

            x_train.append(spectrogram)
            count += 1
    x_train = np.array(x_train)
    # print(x_train.shape)
    # x_train = x_train[..., np.newaxis]  # -> (825, 512, 256, 1)
    return x_train


def interpolate(v1, v2, n=10):
    """
    interpolates between two vectors in latent space
    """
    ratios = np.linspace(0, 1, n)
    int_vecs = list()
    for ratio in ratios:
        v = (1.0 - ratio) * v1 + ratio * v2
        int_vecs.append(v)

    return int_vecs

# print('debugg')
