import numpy as np
from functions import min_max_array_saving, fw_normalise
import os


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
which_set = 'TEST'

"""
original_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_flute_0605_' + which_set + '/'
original_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_string_0605_' + which_set + '/'
original_keyboard_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_keyboard_0605_' + which_set + '/'
"""

original_guitar_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_guitar_1805_' + which_set + '/'
original_organ_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_organ_1805_' + which_set + '/'

"""
normalised_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_flute_0605_' + which_set + '/'
normalised_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_string_0605_' + which_set + '/'
normalised_keyboard_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_keyboard_0605_' + which_set + '/'
"""

normalised_guitar_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_guitar_1805_' + which_set + '/'
normalised_organ_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_organ_1805_' + which_set + '/'

"""
min_max_flute_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/flute_'
min_max_string_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/string_'
min_max_keyboard_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/keyboard_'
"""

min_max_guitar_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/guitar_'
min_max_organ_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/organ_'


try:
    """
    os.mkdir(normalised_flute_path)
    os.mkdir(normalised_string_path)
    os.mkdir(normalised_keyboard_path)
    """
    
    os.mkdir(normalised_guitar_path)
    os.mkdir(normalised_organ_path)
except OSError:
    print("Creation of the directory  failed")
# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZATION & SAVING
# ---------------------------------------------------------------------------------------------------------------------
"""
print('Normalizing flute spectrograms and saving min max values...')
min_max_flute = min_max_array_saving(original_flute_path, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(original_flute_path, normalised_flute_path, min_max_flute)

print('Normalizing string spectrograms and saving min max values...')
min_max_string = min_max_array_saving(original_string_path, min_max_string_path_file)
string_folder_min_max = fw_normalise(original_string_path, normalised_string_path, min_max_string)

print('Normalizing keyboard spectrograms and saving min max values...')
min_max_keyboard = min_max_array_saving(original_keyboard_path, min_max_keyboard_path_file)
keyboard_folder_min_max = fw_normalise(original_keyboard_path, normalised_keyboard_path, min_max_keyboard)
"""

print('Normalizing guitar spectrograms and saving min max values...')
min_max_guitar = min_max_array_saving(original_guitar_path, min_max_guitar_path_file)
guitar_folder_min_max = fw_normalise(original_guitar_path, normalised_guitar_path, min_max_guitar)

print('Normalizing organ spectrograms and saving min max values...')
min_max_organ = min_max_array_saving(original_organ_path, min_max_organ_path_file)
organ_folder_min_max = fw_normalise(original_organ_path, normalised_organ_path, min_max_organ)

"""
np.save(min_max_flute_path_file + "folder_min_max", flute_folder_min_max)
np.save(min_max_string_path_file + "folder_min_max", string_folder_min_max)
np.save(min_max_keyboard_path_file + "folder_min_max", keyboard_folder_min_max)
"""

np.save(min_max_guitar_path_file + "folder_min_max", guitar_folder_min_max)
np.save(min_max_organ_path_file + "folder_min_max", organ_folder_min_max)


print('debag')

