import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import scipy.io.wavfile
from WANDB import config
# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
file_path = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute/flute_acoustic_002-091-050.wav")
storing_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/icqt.wav'
storing_path2 = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/gl_icqt.wav'
SR = config['sample_rate']

'''
y, sr = librosa.load(file_path, res_type='kaiser_fast', mono=True, sr=None)
mel = librosa.feature.melspectrogram(y, sr=SR, n_fft=1024, hop_length=64)
print('mel.shape: ', mel.shape)

fig = plt.figure()
img = plt.imshow(mel, cmap=plt.cm.viridis, origin='lower', aspect='auto')
plt.title('MEL')
plt.colorbar()
plt.show()

y_inv = librosa.feature.inverse.mel_to_audio(mel, sr=SR, n_fft=1024, hop_length=64)
print('inversion done')
scipy.io.wavfile.write(storing_path, SR, y_inv)

fig, ax = plt.subplots(nrows=3)

librosa.display.waveshow(y, sr=SR, ax=ax[0])
ax[0].set(title='ORIGINAL VOCAL')
ax[0].label_outer()

librosa.display.waveshow(y_inv, sr=SR, ax=ax[1])
ax[1].set(title='GENERATED VOCAL')
ax[1].label_outer()

librosa.display.waveshow(y-y_inv, sr=SR, ax=ax[2])
ax[2].set(title='DIFFERENCE')
ax[2].label_outer()

plt.tight_layout()
plt.show()
plt.close()'''


y, sr = librosa.load(file_path, res_type='kaiser_fast', mono=True, sr=None)
cqt = np.abs(librosa.cqt(y, hop_length=128))
print('cqt.shape: ', cqt.shape)

fig = plt.figure()
img = plt.imshow(cqt, cmap=plt.cm.viridis, origin='lower', aspect='auto')
plt.title('CQT')
plt.colorbar()
plt.show()

icqt = librosa.icqt(cqt, hop_length=128)
griffinlimm_cqt = librosa.griffinlim_cqt(cqt, hop_length=128)
scipy.io.wavfile.write(storing_path, SR, icqt)
scipy.io.wavfile.write(storing_path2, SR, griffinlimm_cqt)



fig, ax = plt.subplots(nrows=3)

librosa.display.waveshow(y, sr=SR, ax=ax[0])
ax[0].set(title='ORIGINAL VOCAL')
ax[0].label_outer()

librosa.display.waveshow(icqt, sr=SR, ax=ax[1])
ax[1].set(title='GENERATED VOCAL')
ax[1].label_outer()

librosa.display.waveshow(griffinlimm_cqt, sr=SR, ax=ax[2])
ax[2].set(title='DIFFERENCE')
ax[2].label_outer()

plt.tight_layout()
plt.show()
plt.close()

