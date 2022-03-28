import numpy as np
from functions import min_max_array_saving, fw_normalise


# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------
original_flute_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/features_matching_flute_new_db_formula_VALID/"
original_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/features_matching_vocal_new_db_formula_VALID/"

new_flute_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_VALID/"
new_vocal_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_vocal_VALID/"

min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/flute_"
min_max_vocal_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/vocal_"

# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZATION & SAVING
# ---------------------------------------------------------------------------------------------------------------------
print('Normalizing flute spectrograms and saving min max values...')
min_max_flute = min_max_array_saving(original_flute_path, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(original_flute_path, new_flute_path, min_max_flute)

print('Normalizing vocal spectrograms and saving min max values...')
min_max_vocal = min_max_array_saving(original_vocal_path, min_max_vocal_path_file)
vocal_folder_min_max = fw_normalise(original_vocal_path, new_vocal_path, min_max_vocal)

np.save(min_max_flute_path_file + "folder_min_max", flute_folder_min_max)
np.save(min_max_vocal_path_file + "folder_min_max", vocal_folder_min_max)


print('debag')

