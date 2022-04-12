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


def train_step(self, data):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        '''reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data[0], reconstruction), axis=(1, 2)
            )
        )'''
        reconstruction_loss = K.mean(K.square(data[0] - reconstruction), axis=[1, 2, 3])
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
    }


def train(self, x_train, y_train, train_step):
    for inputs, target in zip(sorted(x_train), sorted(y_train)):
        train_step(inputs, target)
