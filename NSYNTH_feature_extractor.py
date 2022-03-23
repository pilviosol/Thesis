import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from functions import extract_features, feature_calculation

path_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_flute_TEST'
path_matching_vocal = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_vocal_TEST'
path_features_matching_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/features_matching_flute_TEST/'
path_features_matching_vocal = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/features_matching_vocal_TEST/'

'''
try:
    os.mkdir(path_features_matching_flute)
    os.mkdir(path_features_matching_vocal)
except OSError:
    print("Creation of the directory  failed") '''


print("Calculating features for flute (train set).....")
feature_calculation(path_matching_flute, path_features_matching_flute)

print("Calculating features for vocal (train set).....")
feature_calculation(path_matching_vocal, path_features_matching_vocal)

