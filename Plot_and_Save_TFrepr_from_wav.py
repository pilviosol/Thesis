import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib import rc

# ---------------------------------------------------------------------------------------------------------------------
# FONT SETTING
# ---------------------------------------------------------------------------------------------------------------------

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# ---------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------

test_path_file_STFT = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_matching_flute_TRAIN/095_flute_acoustic_007-095-075.wav"
test_path_file_CQT = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/flute_acoustic/flute_acoustic_002-067-075.wav"
saving_path_STFT = "/nas/home/spol/Thesis/AUDIO_TEST/STFT.eps"
saving_path_CQT = "/nas/home/spol/Thesis/AUDIO_TEST/CQT.eps"
saving_path_MEL = "/nas/home/spol/Thesis/AUDIO_TEST/MEL.eps"
saving_path_MAG_PHASE = "/nas/home/spol/Thesis/AUDIO_TEST/MAG_PHASE.eps"

# ---------------------------------------------------------------------------------------------------------------------
# STFT MAG
# ---------------------------------------------------------------------------------------------------------------------

audio, sample_rate = librosa.load(test_path_file_CQT, res_type='kaiser_fast', mono=True, sr=None)
stft_mag = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=128, win_length=2048))
stft_phase = np.angle(librosa.stft(y=audio, n_fft=2048, hop_length=128, win_length=2048))
y_label_list = [1, 512, 1024, 2048, 4096, 8192]

fig, ax = plt.subplots()
img = plt.imshow(10 * np.log10(stft_mag + 1e-2), cmap=plt.cm.viridis, origin='lower', extent=[0, 4, 0, 8192],
                 aspect='auto')
plt.colorbar()
ax.set_yticks([0, 1639, 3278, 4916, 6555, 8192])

ax.set_yticklabels(y_label_list)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Frequency [Hz]', fontsize=14)
plt.savefig(saving_path_STFT, format='eps')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# CQT
# ---------------------------------------------------------------------------------------------------------------------

audio, sample_rate = librosa.load(test_path_file_CQT, res_type='kaiser_fast', mono=True, sr=None)
C = np.abs(
    librosa.cqt(y=audio, sr=sample_rate, hop_length=128, fmin=librosa.note_to_hz('C2'), n_bins=120, bins_per_octave=24))

fig = plt.figure()
img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), hop_length=128, fmin=librosa.note_to_hz('C2'),
                               sr=sample_rate, x_axis='time', y_axis='cqt_note', cmap=plt.cm.viridis,
                               bins_per_octave=24)

plt.colorbar()
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Note', fontsize=15)
plt.savefig(saving_path_CQT, format='eps')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# MEL
# ---------------------------------------------------------------------------------------------------------------------

audio, sample_rate = librosa.load(test_path_file_CQT, res_type='kaiser_fast', mono=True, sr=None)
S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

fig = plt.figure()
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000, cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Frequency [Hz]', fontsize=14)
plt.savefig(saving_path_MEL, format='eps')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# STFT MAG and PHASE
# ---------------------------------------------------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 5))

img1 = ax1.imshow(10 * np.log10(stft_mag + 1e-2), cmap=plt.cm.viridis, origin='lower', extent=[0, 4, 0, 8192],
                  aspect='auto')
ax1.set_xlabel('Time [s]', fontsize=14)
ax1.set_ylabel('Frequency [Hz]', fontsize=14)
ax1.set_yticks([0, 1639, 3278, 4916, 6555, 8192])
ax1.set_yticklabels(y_label_list)

img2 = ax2.imshow(stft_phase, cmap=plt.cm.viridis, origin='lower', extent=[0, 4, 0, 1000], aspect='auto')
ax2.set_xlabel('Time [s]', fontsize=14)
ax2.set_ylabel('Phase (°)', fontsize=14)
plt.savefig(saving_path_MAG_PHASE, format='eps')
plt.show()
