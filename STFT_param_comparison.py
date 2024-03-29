import librosa
import librosa.display
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from functions import resample

SR_16kHz = 16000
violin_4sec_path = "/nas/home/spol/Thesis/violin_4sec/"

nsynth = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST/nsynth-test/audio/bass_electronic_018-023-025.wav"
y, sr = librosa.load(nsynth, sr=None)
stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=128, win_length=1024))


# -------------------------------------------------------------------------------------------------------------------
# RESAMPLING
# -------------------------------------------------------------------------------------------------------------------

y, sr = librosa.load(violin_4sec_path + 'violin_4sec.wav', sr=None)
resample(violin_4sec_path, violin_4sec_path, SR_16kHz)


# -------------------------------------------------------------------------------------------------------------------
# STFT
# -------------------------------------------------------------------------------------------------------------------
y, sr = librosa.load(violin_4sec_path + 'Resampled_violin_4sec.wav', sr=None)
print('y.shape', y.shape)
print('sr', sr)

stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048))
S_n_fft_512_hl_128_wl_512 = np.abs(librosa.stft(y, n_fft=512, hop_length=128, win_length=512))
S_n_fft_512_hl_256_wl_512 = np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=512))

# PLOTS
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('STFT deafault params')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S_n_fft_512_hl_128_wl_512, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('S_n_fft_512_hl_128_wl_512')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S_n_fft_512_hl_256_wl_512, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('S_n_fft_512_hl_256_wl_512')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


# GRIFFIN LIM
y_reconstructed_stft = librosa.griffinlim(stft)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_stft_reconstructed.wav', sr, y_reconstructed_stft)

y_reconstructed_S_n_fft_512_hl_256_wl_512 = librosa.griffinlim(S_n_fft_512_hl_256_wl_512, win_length=512, hop_length=256)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_stft_512_256_512_reconstructed.wav', sr, y_reconstructed_S_n_fft_512_hl_256_wl_512)

y_reconstructed_S_n_fft_512_hl_128_wl_512 = librosa.griffinlim(S_n_fft_512_hl_128_wl_512, win_length=512, hop_length=128)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_stft_512_128_512_reconstructed.wav', sr, y_reconstructed_S_n_fft_512_hl_128_wl_512)




print('debiug')





# -------------------------------------------------------------------------------------------------------------------
# CQT
# -------------------------------------------------------------------------------------------------------------------
y, sr = librosa.load(violin_4sec_path + 'violin_4sec.wav')


C = np.abs(librosa.cqt(y, sr=sr,  n_bins=384, bins_per_octave=48, hop_length=128))
y_hat = librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=48, hop_length=128)
scipy.io.wavfile.write(violin_4sec_path + 'y_hat.wav', sr, y_hat)


'''
C_nb_384_bpo_48_hl_128 = np.abs(librosa.cqt(y, sr=sr,  n_bins=384, bins_per_octave=48, hop_length=128))
C_nb_384_bpo_48_hl_256 = np.abs(librosa.cqt(y, sr=sr,  n_bins=384, bins_per_octave=48, hop_length=256))
C_nb_252_bpo_36_hl_128 = np.abs(librosa.cqt(y, sr=sr,  n_bins=252, bins_per_octave=36, hop_length=128))
C_nb_252_bpo_36_hl_256 = np.abs(librosa.cqt(y, sr=sr,  n_bins=252, bins_per_octave=36, hop_length=256))


y_nb_384_bpo_48_hl_128 = librosa.griffinlim_cqt(C_nb_384_bpo_48_hl_128, sr=sr, bins_per_octave=48, hop_length=128)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_nb_384_bpo_48_hl_128.wav', sr, y_nb_384_bpo_48_hl_128)
y_nb_384_bpo_48_hl_256 = librosa.griffinlim_cqt(C_nb_384_bpo_48_hl_256, sr=sr, bins_per_octave=48, hop_length=256)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_nb_384_bpo_48_hl_256.wav', sr, y_nb_384_bpo_48_hl_256)
y_nb_252_bpo_36_hl_128 = librosa.griffinlim_cqt(C_nb_252_bpo_36_hl_128, sr=sr, bins_per_octave=36, hop_length=128)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_nb_252_bpo_36_hl_128.wav', sr, y_nb_252_bpo_36_hl_128)
y_nb_252_bpo_36_hl_256 = librosa.griffinlim_cqt(C_nb_252_bpo_36_hl_256, sr=sr, bins_per_octave=36, hop_length=256)
scipy.io.wavfile.write(violin_4sec_path + 'Resampled_violin_4sec_nb_252_bpo_36_hl_256.wav', sr, y_nb_252_bpo_36_hl_256)


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
'''
print('debiug')
