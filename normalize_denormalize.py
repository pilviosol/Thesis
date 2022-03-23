from functions import normalise, denormalise
import numpy as np
import matplotlib.pyplot as plt

spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/INFERENCE_flute_features/flute_acoustic_002-079-100_STFTMAG_NEW.npy"

# ---------------------------------------------------------------------------------------------------------------------
# LOAD SPECTROGRAM AND PLOT IT
# ---------------------------------------------------------------------------------------------------------------------
spectrogram = np.load(spectrogram_path)
print('spectrogram.shape: ', spectrogram.shape)

fig = plt.figure()
img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
plt.title('initial spectrogram')
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# NORMALISE SPECTROGRAM AND PLOT IT
# ---------------------------------------------------------------------------------------------------------------------
norm_spectrogram, original_min, original_max = normalise(spectrogram)

fig = plt.figure()
img = plt.imshow(norm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
plt.title('normalised spectrogram')
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# DENORMALISE SPECTROGRAM AND PLOT IT
# ---------------------------------------------------------------------------------------------------------------------
denormalised_spectrogram = denormalise(norm_spectrogram, original_min, original_max)

fig = plt.figure()
img = plt.imshow(denormalised_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
plt.title('denormalised spectrogram')
plt.colorbar()
plt.show()
