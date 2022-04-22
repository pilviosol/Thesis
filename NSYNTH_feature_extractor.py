import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from functions import extract_features, feature_calculation

which_set = 'TRAIN'


path_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/flutes_' + which_set
path_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set + '_SUBSET/strings_' \
                       + which_set
path_features_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                              '_SUBSET/features_flutes_2204_' + which_set + '/'
path_features_matching_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_' + which_set +\
                               '_SUBSET/features_strings_2204_' + which_set + '/'


try:
    os.mkdir(path_features_matching_flute)
    os.mkdir(path_features_matching_string)
except OSError:
    print("Creation of the directory  failed")


print("Calculating features for flute (train set).....")
feature_calculation(path_matching_flute, path_features_matching_flute)

print("Calculating features for string (train set).....")
feature_calculation(path_matching_string, path_features_matching_string)

