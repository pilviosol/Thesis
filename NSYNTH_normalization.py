import numpy as np
from functions import min_max_array_saving, fw_normalise
import os


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
which_set = 'TRAIN'

"""
original_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_flute_0305_' +\
                      which_set + '/'
original_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/NEW_HQ_features_string_0305_' +\
                       which_set + '/' """

original_keyboard_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/features_reducted_keyboards/'
"""
normalised_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_flute_0305_' \
                        + which_set + '/'
normalised_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/FW_normalised_string_0305_' \
                         + which_set + '/' """

normalised_keyboard_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/reducted_keyboards/'

"""
min_max_flute_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/flute_'
min_max_string_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/string_' """

min_max_keyboard_path_file = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/keyboard_'


try:
    # os.mkdir(normalised_flute_path)
    # os.mkdir(normalised_string_path)
    os.mkdir(normalised_keyboard_path)
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
string_folder_min_max = fw_normalise(original_string_path, normalised_string_path, min_max_string) """

print('Normalizing keyboard spectrograms and saving min max values...')
min_max_keyboard = min_max_array_saving(original_keyboard_path, min_max_keyboard_path_file)
keyboard_folder_min_max = fw_normalise(original_keyboard_path, normalised_keyboard_path, min_max_keyboard)

"""
np.save(min_max_flute_path_file + "folder_min_max", flute_folder_min_max)
np.save(min_max_string_path_file + "folder_min_max", string_folder_min_max) """

np.save(min_max_keyboard_path_file + "folder_min_max", keyboard_folder_min_max)


print('debag')

