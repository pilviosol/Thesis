import librosa
import numpy as np
import scipy.io.wavfile
import librosa.display
import matplotlib.pyplot as plt

wav_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_string_TEST/068_string_acoustic_057-068-100.wav'
denorm_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/DIRAC_23-04-2022_14:45/denormalised_spectrogram_TEST' \
              '/GENERATED_normalised_FW_flute_acoustic_002-068-050_STFTMAG_NEW.npy'
N_FFT = 1024
HOP_LENGTH = 128
WIN_LENGTH = 1024


y, sr = librosa.load(wav_path, res_type='kaiser_fast', mono=True, sr=None)

stft = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
stft = stft[0:512, 0:64]

stft_mag = np.abs(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
stft_mag = stft_mag[0:512, 0:64]


stft_phase = np.angle(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
stft_phase = stft_phase[0:512, 0:64]

stftMatrix = stft_mag * np.exp(stft_phase * 1j)

MSE = (np.mean(stft - stftMatrix))**2
print('MSE: ', MSE)


ISTFT = librosa.istft(stftMatrix, hop_length=HOP_LENGTH)
scipy.io.wavfile.write('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/ISTFT.wav', 16000, ISTFT)


GRIFFINLIMM = librosa.griffinlim(stft_mag, hop_length=HOP_LENGTH)
scipy.io.wavfile.write('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/GRIFFINLIMM.wav', 16000, GRIFFINLIMM)


denorm_mag = np.load(denorm_path)
denorm_mag = 10 ** (denorm_mag / 10) - 1e-5
denormMatrix = denorm_mag * np.exp(stft_phase * 1j)
ISTFT_DENORM = librosa.istft(denormMatrix, hop_length=HOP_LENGTH)
scipy.io.wavfile.write('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/ISTF_DENORM.wav', 16000, ISTFT_DENORM)

GRIFFINLIMM_DENORM = librosa.griffinlim(denorm_mag, hop_length=HOP_LENGTH)
scipy.io.wavfile.write('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/GRIFFINLIMM_DENORM.wav', 16000, GRIFFINLIMM_DENORM)

fig, ax = plt.subplots(nrows=4)

librosa.display.waveshow(ISTFT, sr=16000, ax=ax[0])
ax[0].set(title='ISTF')
ax[0].label_outer()

librosa.display.waveshow(GRIFFINLIMM, sr=16000, ax=ax[1])
ax[1].set(title='GRIFFINLIMM')
ax[1].label_outer()


librosa.display.waveshow(ISTFT_DENORM, sr=16000, ax=ax[2])
ax[2].set(title='ISTF_DENORM')
ax[2].label_outer()

librosa.display.waveshow(GRIFFINLIMM_DENORM, sr=16000, ax=ax[3])
ax[3].set(title='GRIFFINLIMM_DENORM')
ax[3].label_outer()
plt.tight_layout()
plt.show()
plt.close()


print('MSE_ISTFT', (np.mean(y[0:8064] - ISTFT))**2)
print('MSE_GRIFFINLIMM', (np.mean(y[0:8064] - GRIFFINLIMM))**2)
print('MSE_ISTFT_DENORM', (np.mean(y[0:8064] - ISTFT_DENORM))**2)
print('MSE_GRIFFINLIMM_DENORM', (np.mean(y[0:8064] - GRIFFINLIMM_DENORM))**2)
