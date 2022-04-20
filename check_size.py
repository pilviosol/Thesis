import numpy as np
import os

path_of_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/NEW_HQ_FW_normalised_flute_VALID/normalised_FW_flute_acoustic_002-094-050_STFTMAG_NEW.npy"
name = path_of_file.split("/")
name = name[-1]


loaded_file = np.load(path_of_file)
print(name, ' Shape: ', loaded_file.shape)

