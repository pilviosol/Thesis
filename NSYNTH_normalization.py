import numpy as np
from functions import min_max_array_saving, fw_normalise
import os


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
which_set = 'TEST'

path_features_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_flute/'
"""
path_features_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_string/'
path_features_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_keyboard/'
path_features_guitar = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_guitar/'
path_features_organ = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_organ/'
"""


normalised_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/NORMALIZED_flute/'
"""
normalised_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/NORMALIZED_string/'
normalised_keyboard_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/NORMALIZED_keyboard/'
normalised_guitar_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/NORMALIZED_guitar/'
normalised_organ_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/NORMALIZED_organ/'
"""

min_max_flute_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/flute_'
"""
min_max_string_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/string_'
min_max_keyboard_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/keyboard_'
min_max_guitar_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/guitar_'
min_max_organ_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/organ_'
"""

try:
    os.mkdir(normalised_flute_path)
    """
    os.mkdir(normalised_string_path)
    os.mkdir(normalised_keyboard_path)
    os.mkdir(normalised_guitar_path)
    os.mkdir(normalised_organ_path)
    """
except OSError:
    print("Creation of the directory  failed")


# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZATION & SAVING
# ---------------------------------------------------------------------------------------------------------------------

print('Normalizing flute spectrograms and saving min max values...')
min_max_flute = min_max_array_saving(path_features_flute, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(path_features_flute, normalised_flute_path, min_max_flute)

"""
print('Normalizing string spectrograms and saving min max values...')
min_max_string = min_max_array_saving(path_features_string, min_max_string_path_file)
string_folder_min_max = fw_normalise(path_features_string, normalised_string_path, min_max_string)

print('Normalizing keyboard spectrograms and saving min max values...')
min_max_keyboard = min_max_array_saving(path_features_keyboard, min_max_keyboard_path_file)
keyboard_folder_min_max = fw_normalise(path_features_keyboard, normalised_keyboard_path, min_max_keyboard)

print('Normalizing guitar spectrograms and saving min max values...')
min_max_guitar = min_max_array_saving(path_features_guitar, min_max_guitar_path_file)
guitar_folder_min_max = fw_normalise(path_features_guitar, normalised_guitar_path, min_max_guitar)

print('Normalizing organ spectrograms and saving min max values...')
min_max_organ = min_max_array_saving(path_features_organ, min_max_organ_path_file)
organ_folder_min_max = fw_normalise(path_features_organ, normalised_organ_path, min_max_organ)
"""

np.save(min_max_flute_path_file + "folder_min_max", flute_folder_min_max)
"""
np.save(min_max_string_path_file + "folder_min_max", string_folder_min_max)
np.save(min_max_keyboard_path_file + "folder_min_max", keyboard_folder_min_max)
np.save(min_max_guitar_path_file + "folder_min_max", guitar_folder_min_max)
np.save(min_max_organ_path_file + "folder_min_max", organ_folder_min_max)
"""

print('debag')

