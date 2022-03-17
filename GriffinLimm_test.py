import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import scipy.io.wavfile

# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
file_path = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/vn_train/AuSep_1_vn_01_Jupiter.wav')
feature_path = '/nas/home/spol/Thesis/URPM_vn_fl/single_feature_vn/AuSep_1_vn_01_Jupiter_STFTMAG.npy_chunk_1.npy'
storing_path = '/nas/home/spol/Thesis/GriffinLimm.wav'

# ---------------------------------------------------------------------------------------------------------------------
# PERFORM LIBROSA STFT, GRIFFINLIM AND ISTFT ON A SONG
# ---------------------------------------------------------------------------------------------------------------------
y, sr = librosa.load(file_path, res_type='kaiser_fast', mono=True)
# Get the magnitude spectrogram
S = np.abs(librosa.stft(y))
# Invert using Griffin-Lim
y_inv = librosa.griffinlim(S)
# Invert without estimating phase
y_istft = librosa.istft(S)


# ---------------------------------------------------------------------------------------------------------------------
# PERFORM LIBROSA STFT, GRIFFINLIM AND ISTFT ON A STORED STFT MAGNITUDE
# ---------------------------------------------------------------------------------------------------------------------
feature_np = np.load(feature_path)
feature_np_squeezed = tf.squeeze(feature_np)
print('feature_np_squeezed.shape: ', feature_np_squeezed.shape)

feature_inv = librosa.griffinlim(feature_np_squeezed.numpy())
scipy.io.wavfile.write(storing_path, 22050, y_inv)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------------------------------------------------
plt.figure()
librosa.display.waveshow(feature_inv, sr=22050, color='b')
plt.show()

