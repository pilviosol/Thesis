import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from matplotlib import rc


path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/normalised_FW_095_flute_acoustic_002-095-025_STFTMAG_NEW.np"
saving_path = "/nas/home/spol/Thesis/plots_thesis/"
spectrogram = np.load(path)


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


y_label_list = [1, 512, 1024, 2048, 4096, 8192]

fig, ax = plt.subplots()
img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 4, 0, 8192],
                 aspect='auto')
plt.colorbar()
ax.set_yticks([0, 1639, 3278, 4916, 6555, 8192])

ax.set_yticklabels(y_label_list)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Frequency [Hz]', fontsize=14)
plt.savefig(saving_path + 'Spectrogram_flute2', format='eps')
plt.show()


