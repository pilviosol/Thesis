from functions import extract_subset, extract_subset_with_path


nsynth_test_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST/nsynth-test/audio/"
nsynth_test_path_subset = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/'
nsynth_train_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN/nsynth-train/audio/"
nsynth_train_path_subset = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/'
nsynth_valid_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID/nsynth-valid/audio/"
nsynth_valid_path_subset = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/'


# extract_subset(nsynth_test_path, nsynth_test_path_subset, 'flute_acoustic')
# extract_subset(nsynth_test_path, nsynth_test_path_subset, 'guitar_acoustic')
# extract_subset(nsynth_train_path, nsynth_train_path_subset, 'flute_acoustic')
# extract_subset(nsynth_train_path, nsynth_train_path_subset, 'organ_electronic')
# extract_subset(nsynth_valid_path, nsynth_valid_path_subset, 'flute_acoustic')
# extract_subset(nsynth_valid_path, nsynth_valid_path_subset, 'organ_electronic')

extract_subset_with_path(nsynth_valid_path, nsynth_valid_path_subset, 'organ_electronic', 'organ_electronic')


