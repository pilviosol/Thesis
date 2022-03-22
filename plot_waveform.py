import librosa
import librosa.display
import matplotlib.pyplot as plt

path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute/021_flute_acoustic_027-021-025.wav"

audio, sample_rate = librosa.load(path, res_type='kaiser_fast', mono=True, sr=None)

fig = plt.figure()
img = librosa.display.waveshow(audio, sr=sample_rate)
plt.show()

fig, ax = plt.subplots(nrows=3)
librosa.display.waveshow(audio, sr=sample_rate, ax=ax[0])
ax[0].set(title='ollare')
ax[0].label_outer()


librosa.display.waveshow(audio*2, sr=sample_rate, ax=ax[1])
ax[1].set(title='ketchup')
ax[1].label_outer()

librosa.display.waveshow(audio*2 - audio, sr=sample_rate, ax=ax[2])
ax[2].set(title='brpow')
ax[2].label_outer()
plt.show()

