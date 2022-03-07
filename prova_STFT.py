import librosa
import librosa.display
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('/nas/home/spol/Thesis/violin_4sec.wav')
print('y.shape', y.shape)
print('sr', sr)

sr_16000 = 16000
y_16k = librosa.resample(y, orig_sr=sr, target_sr=sr_16000)
scipy.io.wavfile.write('/nas/home/spol/Thesis/violin_4sec_16k_resampled.wav', sr_16000, y_16k)

y_hat_16k, sr_16k = librosa.load('/nas/home/spol/Thesis/violin_4sec_16k_resampled.wav', sr=None)
print('y_hat_16k.shape', y_hat_16k.shape)
print('sr_16k', sr_16k)


S_base = np.abs(librosa.stft(y_hat_16k, n_fft=2048, hop_length=512, win_length=2048))
S_n_fft_512_hl_128_wl_512 = np.abs(librosa.stft(y_hat_16k, n_fft=512, hop_length=128, win_length=512))
S_n_fft_512_hl_256_wl_512 = np.abs(librosa.stft(y_hat_16k, n_fft=512, hop_length=256, win_length=512))


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S_base, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('S_base')
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

y_reconstructed_S_base = librosa.griffinlim(S_base,
                                            win_length=2048,
                                            hop_length=512)
y_reconstructed_S_n_fft_512_hl_128_wl_512 = librosa.griffinlim(S_n_fft_512_hl_128_wl_512,
                                                               win_length=512,
                                                               hop_length=128)

scipy.io.wavfile.write('/nas/home/spol/Thesis/violin_4sec_S_base.wav', sr_16000, y_reconstructed_S_base)
scipy.io.wavfile.write('/nas/home/spol/Thesis/violin_4sec_S_n_fft_512_hl_128_wl_512.wav', sr_16000, y_reconstructed_S_n_fft_512_hl_128_wl_512)




print('debiug')
