import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy

# features_matching_flute = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/features_matching_flute/"
features_matching_string = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_FW_normalised_string_TRAIN/"
saving_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/"
flutes = []
# vocals = []
'''
files_dir = pathlib.Path(features_matching_flute)
features_matching_flute = files_dir.iterdir()
for file in sorted(features_matching_flute):
    print(file.name)
    flute = np.load(file)
    flutes.append(flute)
flutes = np.array(flutes) '''

files_dir = pathlib.Path(features_matching_string)
features_matching_string = files_dir.iterdir()
for file in sorted(features_matching_string):
    print(file.name)
    name = file.name[0:-4]

    string = np.load(file)
    fig = plt.figure()
    img = plt.imshow(string, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
    plt.title(name)
    plt.colorbar()
    plt.savefig(saving_path + 'IMAGES_NEW_HQ_FW_normalised_string_TRAIN/' + name)
    # plt.show()
    plt.close()

'''
fig, ax = plt.subplots()
img = librosa.display.specshow(flutes[1000], y_axis='linear', x_axis='time', ax=ax)
ax.set_title('Flute Sample Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()'''

