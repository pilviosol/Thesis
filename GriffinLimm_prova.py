import numpy as np
import tensorflow as tf
import pix2pix_modified
import pathlib
import os
import shutil
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import librosa
import librosa.display
from utils import *
import scipy
import scipy.io.wavfile
from tensorboardX import SummaryWriter
import wandb
from wandb.keras import WandbCallback

file_name = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/vn_train/AuSep_1_vn_01_Jupiter.wav')

'''
y, sr = librosa.load(file_name, res_type='kaiser_fast', mono=True)
# Get the magnitude spectrogram
S = np.abs(librosa.stft(y))
# Invert using Griffin-Lim
y_inv = librosa.griffinlim(S)
# Invert without estimating phase
y_istft = librosa.istft(S)

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
ax[0].set(title='Original', xlabel=None)
ax[0].label_outer()
librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
ax[1].label_outer()
librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
ax[2].set_title('Magnitude-only istft reconstruction')
plt.show()
'''

#/nas/home/spol/Thesis/URPM_vn_fl/vn_train/AuSep_1_vn_01_Jupiter.wav
feature_np = np.load('/nas/home/spol/Thesis/epoch_images/9_new.npy')
feature_np_squeezed = tf.squeeze(feature_np)
print('feature_np_squeezed.shape: ', feature_np_squeezed.shape)

y_inv = librosa.griffinlim(feature_np_squeezed.numpy())
scipy.io.wavfile.write('/nas/home/spol/Thesis/Inverse/inverse.wav', 22050, y_inv)

plt.figure()
librosa.display.waveshow(y_inv, sr=22050, color='b')
plt.show()
