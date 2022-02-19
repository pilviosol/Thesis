import os
import numpy as np
import pathlib
import shutil
import math
from functions import divider_256

# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
CONSTANT_256 = 256  # Rounding of Fs/Hop Length
path_features_vn_train = '/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train'
path_features_fl_train = '/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train'
path_features_vn_test = '/nas/home/spol/Thesis/URPM_vn_fl/features_vn_test'
path_features_fl_test = '/nas/home/spol/Thesis/URPM_vn_fl/features_fl_test'
path_features_vn_train_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train_256/"
path_features_fl_train_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train_256/"
path_features_vn_test_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_test_256/"
path_features_fl_test_256 = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_test_256/"
_SAMPLING_RATE = 22050


# ---------------------------------------------------------------------------------------------------------------------
# CREATION/ REMOVAL OF PATH TO STORE THE FEATURES' CHUNKS
# ---------------------------------------------------------------------------------------------------------------------
try:
    shutil.rmtree(path_features_vn_train_256, ignore_errors=True)
    shutil.rmtree(path_features_fl_train_256, ignore_errors=True)
    shutil.rmtree(path_features_vn_test_256, ignore_errors=True)
    shutil.rmtree(path_features_fl_test_256, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_features_vn_train_256)
    print("Removal of the directory %s failed" % path_features_fl_train_256)
    print("Removal of the directory %s failed" % path_features_vn_test_256)
    print("Removal of the directory %s failed" % path_features_fl_test_256)
else:
    print("Successfully removed the directory %s" % path_features_vn_train_256)
    print("Successfully removed the directory %s" % path_features_fl_train_256)
    print("Successfully removed the directory %s" % path_features_vn_test_256)
    print("Successfully removed the directory %s" % path_features_fl_test_256)


try:
    os.mkdir(path_features_vn_train_256)
    os.mkdir(path_features_fl_train_256)
    os.mkdir(path_features_vn_test_256)
    os.mkdir(path_features_fl_test_256)
except OSError:
    print("Creation of the directory  failed")


# ------------------------------------------------------------------------------------------------------------------
# DIVIDE THE VIOLIN FEATURES IN 256 LONG CHUNKS
# ------------------------------------------------------------------------------------------------------------------
# TRAIN SET
print('Dividing vn train features in 256 long chunks....')
divider_256(path_features_vn_train, path_features_vn_train_256)

# TEST SET
print('Dividing vn test features in 256 long chunks....')
divider_256(path_features_vn_test, path_features_vn_test_256)


# ------------------------------------------------------------------------------------------------------------------
# DIVIDE THE FLUTE FEATURES IN 256 LONG CHUNKS
# ------------------------------------------------------------------------------------------------------------------
# TRAIN SET
print('Dividing fl train features in 256 long chunks....')
divider_256(path_features_fl_train, path_features_fl_train_256)

# TEST SET
print('Dividing fl train features in 256 long chunks....')
divider_256(path_features_fl_test, path_features_fl_test_256)


