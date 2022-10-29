import librosa
import librosa.display
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile

to_be_reversed_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/TO_BE_REVERSED/"
reversed_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/REVERSED/"
to_be_reversed_image_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/TO_BE_REVERSED_IMAGE/"
reversed_image_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/REVERSED_IMAGE/"
wav_plot_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/WAV_PLOTS/"

to_be_reversed_dir = pathlib.Path(to_be_reversed_path)
to_be_reversed = to_be_reversed_dir.iterdir()
for file in sorted(to_be_reversed):
    name = file.name
    name = name[0:-4]
    print(name)

    # AUDIO LOADING
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast', mono=True, sr=None)

    # PLOTTING AUDIO
    fig, ax = plt.subplots(nrows=3)
    librosa.display.waveshow(audio, sr=sample_rate, ax=ax[0])
    ax[0].set(title=name + '_original')
    ax[0].label_outer()


    # STFT
    stft = librosa.stft(y=audio, n_fft=1024, hop_length=128, win_length=1024)
    stft_mag = np.abs(stft)

    # PLOT STFT MAGNITUDE AND SAVE IMAGES
    '''
    fig = plt.figure()
    img = plt.imshow(10 * np.log10(stft_mag + 1e-1), cmap=plt.cm.viridis, origin='lower', extent=[0, 501, 0, 513],
                     aspect='auto')
    plt.title(file.name)
    plt.colorbar()
    plt.savefig(to_be_reversed_image_path + name + '.png') '''

    # iSTFT
    reconstructed = librosa.istft(stft, hop_length=128, win_length=1024)

    # PLOTTING RECONSTRUCTED AUDIO
    librosa.display.waveshow(reconstructed, sr=sample_rate, ax=ax[1])
    ax[1].set(title=name + '_reconstructed')
    ax[1].label_outer()

    # PLOTTING DIFFERENCE AUDIO
    librosa.display.waveshow(audio - reconstructed, sr=sample_rate, ax=ax[2])
    ax[2].set(title=name + '_difference')
    ax[2].label_outer()
    plt.savefig(wav_plot_path + name + '.png')

    # SAVING FILE AUDIO
    # scipy.io.wavfile.write(reversed_path + 'REVERSED_' + name + '.wav', sample_rate, reconstructed)

'''
# -------------------------------------------------------------------------
# Saving spectrograms of reconstructed signals
# -------------------------------------------------------------------------
reversed_dir = pathlib.Path(reversed_path)
reversed = reversed_dir.iterdir()
for file in sorted(reversed):
    name = file.name
    name = name[0:-4]
    print(name)

    # AUDIO RECONSTRUCTED LOADING
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast', mono=True, sr=None)

    # STFT OF RECONSTRUCTED AUDIO
    stft = librosa.stft(y=audio, n_fft=1024, hop_length=128, win_length=1024)
    stft_mag = np.abs(stft)

    # PLOT STFT MAGNITUDE AND SAVE IMAGES OF RECONSTRUCTED AUDIO
    fig = plt.figure()
    img = plt.imshow(10 * np.log10(stft_mag + 1e-1), cmap=plt.cm.viridis, origin='lower', extent=[0, 501, 0, 513],
                     aspect='auto')
    plt.title(file.name)
    plt.colorbar()
    plt.savefig(reversed_image_path + name + '.png') '''

