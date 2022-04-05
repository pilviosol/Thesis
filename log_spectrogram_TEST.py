import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from functions import feature_calculation, min_max_array_saving, fw_normalise

path_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute'
path_features_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute_ls/'
normalised_flute_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/normalised_flute_ls/"
min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute_"


try:
    os.mkdir(path_features_flute)
    os.mkdir(normalised_flute_path)
except OSError:
    print("Creation of the directory  failed")

feature_calculation(path_flute, path_features_flute)

min_max_flute = min_max_array_saving(path_features_flute, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(path_features_flute, normalised_flute_path, min_max_flute)


folder = pathlib.Path(normalised_flute_path).iterdir()

for file in folder:

    norm_spectrogram = np.load(file)
    fig = plt.figure()
    img = plt.imshow(norm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title('NORMALIZED')
    plt.colorbar()
    plt.show()
    plt.close()