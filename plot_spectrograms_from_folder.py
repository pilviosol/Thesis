import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy

features_matching_flute = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/features_matching_flute/"
features_matching_vocal = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/features_matching_vocal/"
flutes = []
vocals = []

files_dir = pathlib.Path(features_matching_flute)
features_matching_flute = files_dir.iterdir()
for file in sorted(features_matching_flute):
    print(file.name)
    flute = np.load(file)
    flutes.append(flute)
flutes = np.array(flutes)

files_dir = pathlib.Path(features_matching_vocal)
features_matching_vocal = files_dir.iterdir()
for file in sorted(features_matching_vocal):
    print(file.name)
    vocal = np.load(file)
    vocals.append(vocal)
vocals = np.array(vocals)


fig, ax = plt.subplots()
img = librosa.display.specshow(flutes[1000], y_axis='linear', x_axis='time', ax=ax)
ax.set_title('Flute Sample Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

