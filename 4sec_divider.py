import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from scipy import io
from scipy.io.wavfile import write



'''

# Facendo delle prove: 
            stft + istft funziona bene, dà un output praticamente identico all'input
            cqt + icqt fa schifo, l'output è irriconoscibile rispetto all'input

file = '/nas/home/spol/Thesis/AuSep_1_vn_01_Jupiter.wav'
y, sr = librosa.load(file, res_type='kaiser_fast', mono=True)
C = librosa.stft(y)
y_hat = librosa.istft(C)

scipy.io.wavfile.write('/nas/home/spol/Thesis/inverse.wav', sr, y_hat)
'''





# DIVIDE THE VIOLIN FEATURES IN 4 SECONDS CHUNKS

CONSTANT_FOUR_SEC = 172   #rounding of Fs/Hop Length

print('Dividing vn features in 4 seconds chunks....')
features_dir_vn = pathlib.Path('URPM_vn_fl/features_vn')
features_in_basepath_vn = features_dir_vn.iterdir()

first_feature = next(features_in_basepath_vn)
print(first_feature)
feature = np.load(first_feature)
print(feature.shape)
feature_cut = feature[0:84,0:172]
print('feature_cut.shape: ', feature_cut.shape)

inverse_4sec = librosa.icqt(feature_cut, sr=22050, hop_length=256, fmin=32.7, filter_scale=0.8, bins_per_octave=48)
inverse_entire = librosa.icqt(feature, sr=22050, hop_length=256, fmin=32.7, filter_scale=0.8, bins_per_octave=48)


scipy.io.wavfile.write('/nas/home/spol/Thesis/inverse.wav', 22050, inverse_entire)
scipy.io.wavfile.write('/nas/home/spol/Thesis/inverse_4sec.wav', 22050, inverse_4sec)


# CQT PLOT OF ENTIRE TRACK
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


# CQT PLOT OF FIRST 4 SEC
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(feature_cut, ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


for item in features_in_basepath_vn:
    print(item.name)
    feature = np.load(item)
    print(feature.shape)
    print('--------------------------------------------')























'''
# DIVIDE THE FLUTE FEATURES IN 4 SECONDS CHUNKS

print('Dividing vn features in 4 seconds chunks....')
features_dir_fl = pathlib.Path('URPM_vn_fl/features_fl')
features_in_basepath_fl = features_dir_fl.iterdir()
print('ollare fl')

for item in features_in_basepath_fl:
    print(item.name) '''
