import os
import numpy as np
import pathlib
import shutil
import math
import librosa
import librosa.display
import scipy.io.wavfile


# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
CONSTANT_256 = 256


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
        stft_mag = np.abs(librosa.stft(y=audio, n_fft=1024, hop_length=128, win_length=1024))

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    # return cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram
    return stft_mag


def feature_calculation(path_songs, store_features_path):
    """

    :param path_songs: folder where all songs to be analyzed are
    :param store_features_path: path where to store all the features
    :return: void
    """
    data_dir = pathlib.Path(path_songs)
    files_in_basepath = data_dir.iterdir()

    for item in files_in_basepath:
        if item.is_file():
            print(item.name)
            name = item.name[0:-4]

            # cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)
            stft_mag = extract_features(item)

            # Saving all features in one folder (.npy format)
            # np.save(store_features_path + name + "_CQT", cqt)
            # np.save(store_features_path + name + "_STFTFULL", stft_full)
            np.save(store_features_path + name + "_STFTMAG", stft_mag)
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


def how_many_pitches(path):
    counts = []
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    count = 0
    temp_pitch = '021'
    for file in sorted(files_in_basepath):
        name = file.name
        pitch = name[0:3]
        if pitch == temp_pitch:
            count += 1
            temp_pitch = pitch
        else:
            counts.append(count)
            temp_pitch = pitch
            count = 0
    return counts
