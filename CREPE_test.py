import crepe
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt

# sr, audio = wavfile.read('AUDIO_TEST/scala.wav')
audio, sr = librosa.load('AUDIO_TEST/scala.wav', res_type='kaiser_fast', mono=True)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)


plt.figure()
plt.scatter(time,frequency)
plt.show()