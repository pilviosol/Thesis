import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from WANDB import config
from tensorflow.keras.optimizers import Adam
import os
import pathlib

LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LATENT_DIM = config['latent_dim']

path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_flute_TRAIN/'
path_features_matching_vocal_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_string_TRAIN/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_VALID/"
path_features_matching_vocal_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_string_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
y_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)
y_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_vocal_val)

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        count = 0
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            '''
            if count % 100 == 0:
                print(count, ", file_name: ", file_name)
                plt.figure()
                plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                           aspect='auto')
                plt.title(str(count))
                plt.colorbar()
                plt.show()
                plt.close()'''
            count += 1
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (4130, 512, 256, 1)
    return x_train


x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
y_train = load_fsdd(y_train_SPECTROGRAMS_PATH)
x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
y_val = load_fsdd(y_val_SPECTROGRAMS_PATH)


# ---------------------------------------------------------------------------------------
# Create a sampling layer
# ---------------------------------------------------------------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ---------------------------------------------------------------------------------------
# Build the encoder
# ---------------------------------------------------------------------------------------


encoder_inputs = keras.Input(shape=(512, 256, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# ---------------------------------------------------------------------------------------
# Build the decoder
# ---------------------------------------------------------------------------------------


latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# ---------------------------------------------------------------------------------------
# Define the VAE as a Model with a custom train_step
# ---------------------------------------------------------------------------------------

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
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


# ---------------------------------------------------------------------------------------
# Train the VAE
# ---------------------------------------------------------------------------------------


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
vae.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))



