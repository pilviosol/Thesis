import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from scipy import io
from scipy.io.wavfile import write
import shutil
import math


path_features_vn_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_256/"
path_features_fl_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_256/"
_SAMPLING_RATE = 22050

try:
    shutil.rmtree(path_features_vn_256, ignore_errors=True)
    shutil.rmtree(path_features_fl_256, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_features_vn_256)
    print("Removal of the directory %s failed" % path_features_fl_256)
else:
    print("Successfully removed the directory %s" % path_features_vn_256)
    print("Successfully removed the directory %s" % path_features_fl_256)


try:
    os.mkdir(path_features_vn_256)
    os.mkdir(path_features_fl_256)
except OSError:
    print("Creation of the directory  failed")

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





# DIVIDE THE VIOLIN FEATURES IN 256 LONG CHUNKS

CONSTANT_FOUR_SEC = 172   #rounding of Fs/Hop Length

print('Dividing vn features in 256 long chunks....')
features_dir_vn = pathlib.Path('URPM_vn_fl/features_vn')
features_in_basepath_vn = features_dir_vn.iterdir()

'''
# TRY ON STFT and ISTFT

first_feature = next(features_in_basepath_vn)
print(first_feature)
feature = np.load(first_feature)
print(feature.shape)


second_feature = next(features_in_basepath_vn)
print(second_feature)
feature = np.load(second_feature)
print(feature.shape)

third_feature = next(features_in_basepath_vn)
print(third_feature)
feature = np.load(third_feature)
print(feature.shape)


fourth_feature = next(features_in_basepath_vn)
print(fourth_feature)
feature = np.load(fourth_feature)
print(feature.shape)

fifth_feature = next(features_in_basepath_vn)
print(fifth_feature)
feature = np.load(fifth_feature)
print(feature.shape)
feature_cut = feature[0:1025,256:512]
print('feature_cut.shape: ', feature_cut.shape)


inverse_256 = librosa.istft(feature_cut, hop_length=512)
inverse_entire = librosa.istft(feature, hop_length=512)


scipy.io.wavfile.write('/nas/home/spol/Thesis/inverse.wav', 22050, inverse_entire)
scipy.io.wavfile.write('/nas/home/spol/Thesis/inverse_4sec.wav', 22050, inverse_256)
'''

'''

# PLOTS

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(feature_cut, ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('256 Power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

'''

for item in features_in_basepath_vn:
    name = item.name
    print(name)
    feature = np.load(item)
    length = feature.shape[1]
    chunks = math.floor(length/256)
    print(chunks)
    for i in range(chunks):
        chunk = feature[0:1025, i * 256 : (i+1) * 256 - 1]
        np.save(path_features_vn_256 + name + "_chunk_" + str(i), chunk)
        print("i: ", i)
    print('--------------------------------------------')







# # DIVIDE THE FLUTE FEATURES IN 256 LONG CHUNKS

print('Dividing vn features in 256 long chunks....')
features_dir_fl = pathlib.Path('URPM_vn_fl/features_fl')
features_in_basepath_fl = features_dir_fl.iterdir()


for item in features_in_basepath_fl:
    name = item.name
    print(name)
    feature = np.load(item)
    length = feature.shape[1]
    chunks = math.floor(length/256)
    print(chunks)
    for i in range(chunks):
        chunk = feature[0:1025, i * 256 : (i+1) * 256 - 1]
        np.save(path_features_fl_256 + name + "_chunk_" + str(i), chunk)
        print("i: ", i)
    print('--------------------------------------------')

