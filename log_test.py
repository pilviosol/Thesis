import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import scipy.io.wavfile
import librosa
from WANDB import config

N_FFT = config['n_fft']
HOP_LENGTH = config['hop_length']
WIN_LENGTH = config['win_length']
SR = config['sample_rate']

file_path = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute/flute_acoustic_002-091-050.wav")
reconstruction_path = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/griffinlimm.wav")


y, sr = librosa.load(file_path, res_type='kaiser_fast', mono=True, sr=None)

stft_mag = np.abs(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
log_spectrogram = 10 * np.log10(stft_mag + 1e-1)
log_spectrogram = log_spectrogram[0:512, 0:256]
spectrogram = 10 ** (log_spectrogram / 10) - 1e-1

fig = plt.figure()
img = plt.imshow(stft_mag, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                 aspect='auto')
plt.title('LOG SPECTROGRAM')
plt.colorbar()
plt.show()

y_inv = librosa.griffinlim(spectrogram)
scipy.io.wavfile.write(reconstruction_path, 16000, y_inv)


fig, ax = plt.subplots(nrows=3)

librosa.display.waveshow(y[0:32640], sr=SR, ax=ax[0])
ax[0].set(title='ORIGINAL VOCAL')
ax[0].label_outer()

librosa.display.waveshow(y_inv[0:32640], sr=SR, ax=ax[1])
ax[1].set(title='GENERATED VOCAL')
ax[1].label_outer()

librosa.display.waveshow(y[0:32640] - y_inv[0:32640], sr=SR, ax=ax[2])
ax[2].set(title='DIFFERENCE')
ax[2].label_outer()

plt.tight_layout()
plt.show()
plt.close()

print(np.mean((y[0:32640] - y_inv[0:32640]) ** 2))

