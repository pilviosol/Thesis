import librosa
import librosa.display
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('/nas/home/spol/Thesis/scala_sustained.wav')
print('y.shape', y.shape)
print('original sr', sr)

new_sr = 8000
y_8k = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
scipy.io.wavfile.write('/nas/home/spol/Thesis/resampled_scala_sustained.wav', new_sr, y_8k)

y_hat, sr_new = librosa.load('/nas/home/spol/Thesis/resampled_scala_sustained.wav', sr=None)
print('y_hat.shape', y_hat.shape)
print('new sr', sr_new)

'''
S1 = np.abs(librosa.stft(y))


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S1,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
print('debiug')


y, sr = librosa.load('/nas/home/spol/Thesis/scala_sustained.wav')
S = np.abs(librosa.stft(y, n_fft=512, hop_length=1024))


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
print('debiug')
'''
