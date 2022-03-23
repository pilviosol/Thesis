import matplotlib.pyplot as plt
import numpy as np

#path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/normalised_features_matching_flute_new_db_formula/normalised_new_038_flute_acoustic_003-038-025_STFTMAG_NEW.npy"
path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/normalised_features_matching_vocal_new_db_formula/normalised_new_039_vocal_synthetic_001-039-025_STFTMAG_NEW.npy"
spectrogram = np.load(path)

fig = plt.figure()
img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                 aspect='auto')
plt.title('spectrogram')
plt.colorbar()
plt.show()




