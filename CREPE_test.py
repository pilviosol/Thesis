import crepe
import librosa
import matplotlib.pyplot as plt

audio, sr = librosa.load('AUDIO_TEST/scala.wav', res_type='kaiser_fast', mono=True)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)


plt.figure()
plt.scatter(time,frequency)
plt.show()