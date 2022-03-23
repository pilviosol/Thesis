from functions import feature_calculation
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math

test_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/INFERENCE_flute/"
test_path_feature = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/INFERENCE_flute_features/"
test_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/INFERENCE_flute/flute_acoustic_002-079-100.wav"


audio, sample_rate = librosa.load(test_path_file, res_type='kaiser_fast', mono=True, sr=None)
stft_mag = np.abs(librosa.stft(y=audio, n_fft=1024, hop_length=128, win_length=1024))
# log_spectrogram = librosa.amplitude_to_db(stft_mag)

fig = plt.figure()
img = plt.imshow(10*np.log10(stft_mag + 1e-1), cmap=plt.cm.viridis, origin='lower', extent=[0,501,0,513], aspect='auto')
plt.title('Power spectrogram FROM FILE')
plt.colorbar()
plt.show()
# -------------------------------------------------------------------------------------------------
spectr = np.load("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/INFERENCE_flute_features/flute_acoustic_002-079-100_STFTMAG_NEW.npy")

fig = plt.figure()
img = plt.imshow(10*np.log10(spectr + 1e-1), cmap=plt.cm.viridis, origin='lower', extent=[0,501,0,513], aspect='auto')
plt.title('Power spectrogram FROM FILE')
plt.colorbar()
plt.show()



