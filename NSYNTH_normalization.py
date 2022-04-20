import numpy as np
from functions import min_max_array_saving, fw_normalise
import os


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
which_set = 'TRAIN'


original_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_matching_flute_2004_' +\
                      which_set + '/'
original_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_matching_string_2004_' +\
                       which_set + '/'

normalised_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_FW_normalised_flute_' \
                        + which_set + '/'
normalised_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_FW_normalised_string_' \
                         + which_set + '/'
min_max_flute_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/flute_'
min_max_string_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/string_'


try:
    os.mkdir(normalised_flute_path)
    os.mkdir(normalised_string_path)
except OSError:
    print("Creation of the directory  failed")
# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZATION & SAVING
# ---------------------------------------------------------------------------------------------------------------------
print('Normalizing flute spectrograms and saving min max values...')
min_max_flute = min_max_array_saving(original_flute_path, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(original_flute_path, normalised_flute_path, min_max_flute)

print('Normalizing string spectrograms and saving min max values...')
min_max_string = min_max_array_saving(original_string_path, min_max_string_path_file)
string_folder_min_max = fw_normalise(original_string_path, normalised_string_path, min_max_string)

np.save(min_max_flute_path_file + "folder_min_max", flute_folder_min_max)
np.save(min_max_string_path_file + "folder_min_max", string_folder_min_max)


print('debag')

