import crepe
import librosa
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------------------------------
# LOAD A WAV FILE AND PERFORM CREPE PREDICTION
# ---------------------------------------------------------------------------------------------------------------------
audio, sr = librosa.load('AUDIO_TEST/scala.wav', res_type='kaiser_fast', mono=True)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT PREDICTION
# ---------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.scatter(time, frequency)
plt.show()
