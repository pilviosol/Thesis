'''

# PLOTS

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(feature_cut, ref=np.max),
                                sr=22050, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('256 Power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()








# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------

flutes_path = pathlib.Path(features_matching_flute_TEST).iterdir()
vocals_path = pathlib.Path(features_matching_vocal_TEST).iterdir()

for flute, vocal in zip(sorted(flutes_path), sorted(vocals_path)):
    flute_name = flute.name
    print('flute_name[19:-16]: ', flute_name[19:-16])
    flute_spectrogram = np.load(flute)
    flute_spectrogram = flute_spectrogram[0:512, 0:256]
    vocal_name = vocal.name
    vocal_spectrogram = np.load(vocal)
    vocal_spectrogram = vocal_spectrogram[0:512, 0:256]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # FLUTE SPECTROGRAMS
    img1 = ax1.imshow(flute_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax1.set_title('flute_' + flute_name[19:-16])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    # CORRESPONDING VOCAL SPECTROGRAMS (WHAT WE WOULD WANT AS OUTPUT)
    img2 = ax2.imshow(vocal_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax2.set_title('vocal_' + vocal_name[20:-16])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)
    plt.savefig(Figures_TEST + 'input-expected_output/' + flute_name[19:-16])
    # plt.show()

    plt.close()

print('PLOT THE SPECTROGRAMS..........ok')







print('debug')




'''