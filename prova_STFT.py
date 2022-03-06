import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('/nas/home/spol/Thesis/scala_sustained.wav')
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