import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa


path_images = "/nas/home/spol/Thesis/URPM_vn_fl/images/"
path_features_vn = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn/"
path_features_fl = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl/"
_SAMPLING_RATE = 48000
print('ollare')


try:
    os.mkdir(path_features_vn)
    os.mkdir(path_features_fl)
except OSError:
    print("Creation of the directory  failed")



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
        cqt = librosa.cqt(y=audio, sr=sample_rate, hop_length=256, fmin=32.7, filter_scale=0.8, bins_per_octave=48)
        stft_mag = np.abs(librosa.stft(y=audio, hop_length=256))
        stft_mag_real = stft_mag.real
        stft_mag_imag = stft_mag.imag
        stft_phase = np.angle(librosa.stft(y=audio, hop_length=256))
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,
                                                         n_fft=2048, hop_length=256,
                                                         n_mels=128)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return cqt, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram


# VIOLIN

print("Calculating features for violin.....")

data_dir_vn = pathlib.Path('URPM_vn_fl/vn_train')
files_in_basepath_vn = data_dir_vn.iterdir()

for item in files_in_basepath_vn:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_vn + name + "_CQT", cqt)
        np.save(path_features_vn + name + "_STFTMAG", stft_mag)
        np.save(path_features_vn + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_vn + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_vn + name + "_STFTPHASE", stft_phase)
        np.save(path_features_vn + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')





# FLUTE

print("Calculating features for flute.....")

data_dir_fl = pathlib.Path('URPM_vn_fl/fl_train')
files_in_basepath_fl = data_dir_fl.iterdir()

for item in files_in_basepath_fl:
    if item.is_file():
        print(item.name)
        name = item.name[0:-4]

        cqt, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

        # Saving all features in one folder (.npy format)
        np.save(path_features_fl + name + "_CQT", cqt)
        np.save(path_features_fl + name + "_STFTMAG", stft_mag)
        np.save(path_features_fl + name + "_STFTMAG_REAL", stft_mag_real)
        np.save(path_features_fl + name + "_STFTMAG_IMAG", stft_mag_imag)
        np.save(path_features_fl + name + "_STFTPHASE", stft_phase)
        np.save(path_features_fl + name + "_MEL", mel_spectrogram)

    else:
        print('That is not a file')


# SHOW A SAMPLE IMAGE OF MAGNITUDE AND PHASE OF A STFT
stft_mag = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn/AuSep_1_vn_35_Rondeau_STFTMAG.npy')
stft_phase = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn/AuSep_1_vn_35_Rondeau_STFTPHASE.npy')


plt.figure()
plt.subplot(2,1,1)
plt.imshow(20*np.log10(stft_mag + 1E-5), aspect='auto')

plt.subplot(2,1,2)
plt.imshow(np.unwrap(stft_phase), aspect='auto')
plt.show()


