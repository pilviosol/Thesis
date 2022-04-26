import numpy as np
import pathlib
import matplotlib.pyplot as plt


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_flutes_TRAIN/'
y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train).iterdir()
savefig_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/IMAGES_FW_normalised_flutes_TRAIN/'


for file in sorted(y_train_SPECTROGRAMS_PATH):
    name = file.name
    print(name[0:-16])
    spectrogram = np.load(file)
    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title(name[0:-16])
    plt.colorbar()
    plt.savefig(savefig_path + name[0:-16])
    # plt.show()
    plt.close()



