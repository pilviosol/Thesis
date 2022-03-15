import numpy as np
from functions import normalise_set_and_save_min_max

original_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/test/"
new_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/test_normalized/"
min_max_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/min_max.npy"

min_max = normalise_set_and_save_min_max(original_path, new_path)

np.save(min_max_path_file, min_max)

a = np.load('/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/test_normalized/normalised_106_flute_acoustic_016-106-127_STFTMAG.npy')
mm = np.load(min_max_path_file)

print('debag')