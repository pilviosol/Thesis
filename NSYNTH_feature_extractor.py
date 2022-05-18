import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from functions import extract_features, feature_calculation

which_set = 'TEST'

"""
path_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_flute_' + which_set
path_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_string_' + which_set
path_matching_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_keyboard_' + which_set
"""

path_matching_guitar = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_guitar_' + which_set
path_matching_organ = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_organ_' + which_set

"""
path_features_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                              '_SUBSET/NEW_HQ_features_flute_0605_' + which_set + '/'
path_features_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/NEW_HQ_features_string_0605_' + which_set + '/'
path_features_matching_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/NEW_HQ_features_keyboard_0605_' + which_set + '/' """

path_features_matching_guitar = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/NEW_HQ_features_guitar_1805_' + which_set + '/'
path_features_matching_organ = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/NEW_HQ_features_organ_1805_' + which_set + '/'


try:
    """
    os.mkdir(path_features_matching_flute)
    os.mkdir(path_features_matching_string)
    os.mkdir(path_features_matching_keyboard)
    """
    os.mkdir(path_features_matching_guitar)
    os.mkdir(path_features_matching_organ)
except OSError:
    print("Creation of the directory  failed")


"""
print("Calculating features for flute (test set).....")
feature_calculation(path_matching_flute, path_features_matching_flute)

print("Calculating features for string (test set).....")
feature_calculation(path_matching_string, path_features_matching_string)

print("Calculating features for keyboard (test set).....")
feature_calculation(path_matching_keyboard, path_features_matching_keyboard)
"""

print("Calculating features for guitar (train set).....")
feature_calculation(path_matching_guitar, path_features_matching_guitar)

print("Calculating features for organ (train set).....")
feature_calculation(path_matching_organ, path_features_matching_organ)

