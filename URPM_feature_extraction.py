import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa
import shutil

'''
PROVARE IL GRIFFIN LIMM ALGORITHM PER I CAZZI MIEI BELLA
'''




# DEFINITION OF PATHS

path_features_vn_train = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/"
path_features_fl_train = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train/"
path_features_vn_test = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_test/"
path_features_fl_test = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_test/"
_SAMPLING_RATE = 22050
print('ollare')



try:
    shutil.rmtree(path_features_vn_train, ignore_errors=True)
    shutil.rmtree(path_features_fl_train, ignore_errors=True)
    shutil.rmtree(path_features_vn_test, ignore_errors=True)
    shutil.rmtree(path_features_fl_test, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_features_vn_train)
    print("Removal of the directory %s failed" % path_features_fl_train)
    print("Removal of the directory %s failed" % path_features_vn_test)
    print("Removal of the directory %s failed" % path_features_fl_test)
else:
    print("Successfully removed the directory %s" % path_features_vn_train)
    print("Successfully removed the directory %s" % path_features_fl_train)
    print("Successfully removed the directory %s" % path_features_vn_test)
    print("Successfully removed the directory %s" % path_features_fl_test)


try:
    os.mkdir(path_features_vn_train)
    os.mkdir(path_features_fl_train)
    os.mkdir(path_features_vn_test)
    os.mkdir(path_features_fl_test)
except OSError:
    print("Creation of the directory  failed")


# ------------------------------------------------------------------------------------------------------------------
# DEFINITION OF FUNCTION EXTRACT_FEATURE

def extract_features(file_name):
    '''

    :param file_name: file to be analyzed
    :return:
    cqt: constnt Q transform
    stft_mag: magnitude of STFT
    stft_mag_real: real part of stft magnitude
    stft_mag_imag: imaginary part of stft magnitude
    stft_phase: phase of stft
    mel_spectrogram: spectrogram using mel coefficients

    '''
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', mono=True)
        print('sample_rate: ', sample_rate)
        cqt = librosa.cqt(y=audio, sr=sample_rate, hop_length=512, fmin=32.7, filter_scale=0.8,
                          bins_per_octave=48, n_bins=384)
        stft_full = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        stft_mag = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=512))
        stft_mag_real = stft_mag.real
        stft_mag_imag = stft_mag.imag
        stft_phase = np.angle(librosa.stft(y=audio, hop_length=512))
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,
                                                         n_fft=2048, hop_length=512,
                                                         n_mels=128)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram


# ------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FOR VIOLIN

# VIOLIN

# TRAIN SET

print("Calculating features for violin (train set).....")

data_dir_vn = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/vn_train')
files_in_basepath_vn = data_dir_vn.iterdir()

for item in files_in_basepath_vn:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_vn_train + name + "_CQT", cqt)
        np.save(path_features_vn_train + name + "_STFTFULL", stft_full)
        np.save(path_features_vn_train + name + "_STFTMAG", stft_mag)
        np.save(path_features_vn_train + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_vn_train + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_vn_train + name + "_STFTPHASE", stft_phase)
        np.save(path_features_vn_train + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')


# TEST SET

print("Calculating features for violin (test set).....")

data_dir_vn = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/vn_test')
files_in_basepath_vn = data_dir_vn.iterdir()

for item in files_in_basepath_vn:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_vn_test + name + "_CQT", cqt)
        np.save(path_features_vn_test + name + "_STFTFULL", stft_full)
        np.save(path_features_vn_test + name + "_STFTMAG", stft_mag)
        np.save(path_features_vn_test + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_vn_test + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_vn_test + name + "_STFTPHASE", stft_phase)
        np.save(path_features_vn_test + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')

# ------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FOR FLUTE

# FLUTE

# TRAIN SET


print("Calculating features for flute (train set).....")

data_dir_fl = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/fl_train')
files_in_basepath_fl = data_dir_fl.iterdir()

for item in files_in_basepath_fl:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_fl_train + name + "_CQT", cqt)
        np.save(path_features_fl_train + name + "_STFTFULL", stft_full)
        np.save(path_features_fl_train + name + "_STFTMAG", stft_mag)
        np.save(path_features_fl_train + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_fl_train + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_fl_train + name + "_STFTPHASE", stft_phase)
        np.save(path_features_fl_train + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')


# TEST SET

print("Calculating features for flute (test set).....")

data_dir_fl = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/fl_test')
files_in_basepath_fl = data_dir_fl.iterdir()

for item in files_in_basepath_fl:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_full, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_fl_test + name + "_CQT", cqt)
        np.save(path_features_fl_test + name + "_STFTFULL", stft_full)
        np.save(path_features_fl_test + name + "_STFTMAG", stft_mag)
        np.save(path_features_fl_test + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_fl_test + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_fl_test + name + "_STFTPHASE", stft_phase)
        np.save(path_features_fl_test + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')


# ------------------------------------------------------------------------------------------------------------------


# SHOW A SAMPLE IMAGE OF MAGNITUDE AND PHASE OF A STFT
stft_mag = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/AuSep_1_vn_35_Rondeau_STFTMAG.npy')
stft_phase = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/AuSep_1_vn_35_Rondeau_STFTPHASE.npy')


plt.figure()
plt.subplot(2,1,1)
plt.imshow(20*np.log10(stft_mag + 1E-5), aspect='auto')

plt.subplot(2,1,2)
plt.imshow(np.unwrap(stft_phase), aspect='auto')
plt.show()



