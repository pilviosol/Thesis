import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from functions import extract_features, feature_calculation


# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
path_vn_train_wav_folder = '/nas/home/spol/Thesis/URPM_vn_fl/vn_train'
path_fl_train_wav_folder = '/nas/home/spol/Thesis/URPM_vn_fl/fl_train'
path_vn_test_wav_folder = '/nas/home/spol/Thesis/URPM_vn_fl/vn_test'
path_fl_test_wav_folder = '/nas/home/spol/Thesis/URPM_vn_fl/fl_test'


path_features_vn_train = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/"
path_features_fl_train = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train/"
path_features_vn_test = "/nas/home/spol/Thesis/URPM_vn_fl/features_vn_test/"
path_features_fl_test = "/nas/home/spol/Thesis/URPM_vn_fl/features_fl_test/"
_SAMPLING_RATE = 22050


# ---------------------------------------------------------------------------------------------------------------------
# CREATION/ REMOVAL OF PATH TO STORE THE FEATURES' CHUNKS
# ---------------------------------------------------------------------------------------------------------------------
try:
    shutil.rmtree(path_features_vn_train, ignore_errors=True)
    shutil.rmtree(path_features_fl_train, ignore_errors=True)
    shutil.rmtree(path_features_vn_test, ignore_errors=True)
    shutil.rmtree(path_features_fl_test, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_features_vn_train)
    print("Removal of the directory %s failed" % path_features_fl_train)
    print("Removal of the directory %s failed" % path_features_vn_test)
    print("Removal of the directory %s failed" % path_features_fl_test)
else:
    print("Successfully removed the directory %s" % path_features_vn_train)
    print("Successfully removed the directory %s" % path_features_fl_train)
    print("Successfully removed the directory %s" % path_features_vn_test)
    print("Successfully removed the directory %s" % path_features_fl_test)


try:
    os.mkdir(path_features_vn_train)
    os.mkdir(path_features_fl_train)
    os.mkdir(path_features_vn_test)
    os.mkdir(path_features_fl_test)
except OSError:
    print("Creation of the directory  failed")


# ------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FOR VIOLIN
# ------------------------------------------------------------------------------------------------------------------

# TRAIN SET
print("Calculating features for violin (train set).....")
feature_calculation(path_vn_train_wav_folder, path_features_vn_train)

# TEST SET
print("Calculating features for violin (test set).....")
feature_calculation(path_vn_test_wav_folder, path_features_vn_test)

# ------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FOR FLUTE
# ------------------------------------------------------------------------------------------------------------------

# TRAIN SET
print("Calculating features for flute (train set).....")
feature_calculation(path_fl_train_wav_folder, path_features_fl_train)

# TEST SET
print("Calculating features for flute (test set).....")
feature_calculation(path_fl_test_wav_folder, path_features_fl_test)


# ------------------------------------------------------------------------------------------------------------------
# SHOW A SAMPLE IMAGE OF MAGNITUDE AND PHASE OF A STFT
# ------------------------------------------------------------------------------------------------------------------
stft_mag = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/AuSep_1_vn_35_Rondeau_STFTMAG.npy')
stft_phase = np.load('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train/AuSep_1_vn_35_Rondeau_STFTPHASE.npy')

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(20*np.log10(stft_mag + 1E-5), aspect='auto')

plt.subplot(2, 1, 2)
plt.imshow(np.unwrap(stft_phase), aspect='auto')
plt.show()

