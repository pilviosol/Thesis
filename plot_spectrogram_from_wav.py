from functions import feature_calculation
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math

test_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute_features/"
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/testing_flute/021_flute_acoustic_027-021-025.wav"


audio, sample_rate = librosa.load(test_path_file, res_type='kaiser_fast', mono=True, sr=None)
stft_mag = np.abs(librosa.stft(y=audio, n_fft=1024, hop_length=128, win_length=1024))
# log_spectrogram = librosa.amplitude_to_db(stft_mag)

fig = plt.figure()
img = plt.imshow(10*np.log10(stft_mag + 1e-1), cmap=plt.cm.viridis, origin='lower', extent=[0,501,0,513], aspect='auto')
plt.title('Power spectrogram FROM FILE')
plt.colorbar()
plt.show()
# -------------------------------------------------------------------------------------------------





