import numpy as np
from functions import normalise_set_and_save_min_max


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
original_flute_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/features_matching_flute_nodb_VALID/"
original_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/features_matching_vocal_nodb_VALID/"

new_flute_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/normalised_features_matching_flute_nodb_VALID/"
new_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/normalised_features_matching_vocal_nodb_VALID/"

min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/flute_min_max_nodb_VALID.npy"
min_max_vocal_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/vocal_min_max_nodb_VALID.npy"

# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZATION & SAVING
# ---------------------------------------------------------------------------------------------------------------------
print('Normalizing flute spectrograms and saving min max values...')
min_max_flute = normalise_set_and_save_min_max(original_flute_path, new_flute_path)
print('Normalizing vocal spectrograms and saving min max values...')
min_max_vocal = normalise_set_and_save_min_max(original_vocal_path, new_vocal_path)

np.save(min_max_flute_path_file, min_max_flute)
np.save(min_max_vocal_path_file, min_max_vocal)


print('debag')