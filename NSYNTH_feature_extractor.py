import os
from functions import feature_calculation

which_set = 'TEST'

path_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/WAV_flute/'
"""
path_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/WAV_string/'
path_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/WAV_keyboard/'
path_guitar = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/WAV_guitar/'
path_organ = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/WAV_organ/'
"""

path_features_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_flute/'
"""
path_features_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_string/'
path_features_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_keyboard/'
path_features_guitar = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_guitar/'
path_features_organ = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/07062022/FEATURES_organ/'
"""

try:
    os.mkdir(path_features_flute)
    """
    os.mkdir(path_features_string)
    os.mkdir(path_features_keyboard)
    os.mkdir(path_features_guitar)
    os.mkdir(path_features_organ)
    """
except OSError:
    print("Creation of the directory  failed")

print("Calculating features for flute (" + which_set + " set).....")
feature_calculation(path_flute, path_features_flute)

"""
print("Calculating features for string (" + which_set + " set).....")
feature_calculation(path_string, path_features_string)

print("Calculating features for organ (" + which_set + " set).....")
feature_calculation(path_keyboard, path_features_keyboard)

print("Calculating features for guitar (" + which_set + " set).....")
feature_calculation(path_guitar, path_features_guitar)

print("Calculating features for organ (" + which_set + " set).....")
feature_calculation(path_organ, path_features_organ)
"""
