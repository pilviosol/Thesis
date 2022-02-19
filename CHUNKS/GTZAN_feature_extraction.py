import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa
import shutil

path_images = "/nas/home/spol/Thesis/GTZAN/images/"
path_features = "/nas/home/spol/Thesis/GTZAN/features/"
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
_SAMPLING_RATE = 22000
print('ollare')

try:
    shutil.rmtree(path_images, ignore_errors=True)
    shutil.rmtree(path_features, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_images)
else:
    print("Successfully removed the directory %s" % path_images)

for genre in genres:
    print(path_images + genre)
    try:
        os.makedirs(path_images + genre + "/train")
        os.makedirs(path_images + genre + "/test")
        os.makedirs(path_features + genre + "/train")
        os.makedirs(path_features + genre + "/test")
    except OSError:
        print("Creation of the directory %s failed" % (path_images + genre))
    else:
        print("Successfully created the directory %s" % (path_images + genre))


def extract_features(file_name):
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


for genre in genres:
    data_dir = pathlib.Path('GTZAN/genres_original' + '/' + genre)
    files_in_basepath = data_dir.iterdir()
    count = 0
    for item in files_in_basepath:
        if item.is_file():
            print(item.name)
            name = item.name[0:-4]

            cqt, stft_mag, stft_mag_real, stft_mag_imag, stft_phase, mel_spectrogram = extract_features(item)

            fig_cqt, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                                           sr=_SAMPLING_RATE, ax=ax)

            fig_stftmag, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(stft_mag, ref=np.max),
                                           sr=_SAMPLING_RATE, ax=ax)

            fig_stftphase, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(stft_phase, ref=np.max),
                                           sr=_SAMPLING_RATE, ax=ax)

            if count < 80:
                fig_cqt.savefig(path_images + genre + "/train/" + name + "_CQT.jpg", bbox_inches='tight', pad_inches=-0.1)
                fig_stftmag.savefig(path_images + genre + "/train/" + name + "_STFTMAG.jpg", bbox_inches='tight',
                                    pad_inches=-0.1)
                fig_stftphase.savefig(path_images + genre + "/train/" + name + "_STFTPHASE.jpg", bbox_inches='tight',
                                      pad_inches=-0.1)
                np.save(path_features + genre + "/train/" + name + "_CQT", cqt)
                np.save(path_features + genre + "/train/" + name + "_STFTMAG", stft_mag)
                np.save(path_features + genre + "/train/" + name + "_STFTMAG_REAL", stft_mag_real)
                np.save(path_features + genre + "/train/" + name + "_STFTMAG_REAL", stft_mag_imag)
                np.save(path_features + genre + "/train/" + name + "_STFTPHASE", stft_phase)
                np.save(path_features + genre + "/train/" + name + "_MEL", mel_spectrogram)
            else:
                fig_cqt.savefig(path_images + genre + "/test/" + name + "_CQT.jpg", bbox_inches='tight', pad_inches=-0.1)
                fig_stftmag.savefig(path_images + genre + "/test/" + name + "_STFTMAG.jpg", bbox_inches='tight',
                                    pad_inches=-0.1)
                fig_stftphase.savefig(path_images + genre + "/test/" + name + "_STFTPHASE.jpg", bbox_inches='tight',
                                      pad_inches=-0.1)
                np.save(path_features + genre + "/test/" + name + "_CQT", cqt)
                np.save(path_features + genre + "/test/" + name + "_STFTMAG", stft_mag)
                np.save(path_features + genre + "/test/" + name + "_STFTMAG_REAL", stft_mag_real)
                np.save(path_features + genre + "/test/" + name + "_STFTMAG_REAL", stft_mag_imag)
                np.save(path_features + genre + "/test/" + name + "_STFTPHASE", stft_phase)
                np.save(path_features + genre + "/test/" + name + "_MEL", mel_spectrogram)
            plt.close(fig_cqt)
            plt.close(fig_stftmag)
            plt.close(fig_stftphase)
        else:
            print('That is not a file')
        count += 1
