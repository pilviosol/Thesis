import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from functions import extract_features, feature_calculation

which_set = 'TRAIN'


path_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_flute_' + which_set
path_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/matching_string_' + which_set
path_matching_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/keyboard_acoustic'

path_features_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                              '_SUBSET/NEW_HQ_features_flute_0305_' + which_set + '/'
path_features_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/NEW_HQ_features_string_0305_' + which_set + '/'
path_features_matching_keyboard = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/reducted_keyboards/'


try:
    # os.mkdir(path_features_matching_flute)
    # os.mkdir(path_features_matching_string)
    os.mkdir(path_features_matching_keyboard)
except OSError:
    print("Creation of the directory  failed")

'''
print("Calculating features for flute (train set).....")
feature_calculation(path_matching_flute, path_features_matching_flute)

print("Calculating features for flute (train set).....")
feature_calculation(path_matching_string, path_features_matching_string) '''

print("Calculating features for keyboard (train set).....")
feature_calculation(path_matching_keyboard, path_features_matching_keyboard)

