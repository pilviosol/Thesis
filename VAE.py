import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from WANDB import config
from tensorflow.keras.optimizers import Adam
import os
import pathlib
import pickle

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_kl_loss = tf.keras.metrics.Mean(name="train_kl_loss")
train_reconstruction_loss = tf.keras.metrics.Mean(name="train_reconstruction_loss")


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

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_val.shape: ', x_val.shape)
print('y_val.shape: ', y_val.shape)

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

x = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding="same", name="encoder_conv_layer_1")(encoder_inputs)
x = layers.ReLU(name=f"encoder_relu_1")(x)
x = layers.BatchNormalization(name="encoder_bn_1")(x)

x = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", name="encoder_conv_layer_2")(x)
x = layers.ReLU(name=f"encoder_relu_2")(x)
x = layers.BatchNormalization(name="encoder_bn_2")(x)

x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", name="encoder_conv_layer_3")(x)
x = layers.ReLU(name=f"encoder_relu_3")(x)
x = layers.BatchNormalization(name="encoder_bn_3")(x)

x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", name="encoder_conv_layer_4")(x)
x = layers.ReLU(name=f"encoder_relu_4")(x)
x = layers.BatchNormalization(name="encoder_bn_4")(x)

x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 1), padding="same", name="encoder_conv_layer_5")(x)
x = layers.ReLU(name=f"encoder_relu_5")(x)
x = layers.BatchNormalization(name="encoder_bn_5")(x)

x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# ---------------------------------------------------------------------------------------
# Build the decoder
# ---------------------------------------------------------------------------------------


latent_inputs = keras.Input(shape=(LATENT_DIM,), name="decoder_input")
x = layers.Dense(LATENT_DIM * LATENT_DIM * 32, activation="relu")(latent_inputs)
x = layers.Reshape((LATENT_DIM, LATENT_DIM, 32))(x)

x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 1), padding="same", name="decoder_conv_transpose_layer_1")(x)
x = layers.ReLU(name="decoder_relu_1")(x)
x = layers.BatchNormalization(name="decoder_bn_1")(x)

x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", name="decoder_conv_transpose_layer_2")(x)
x = layers.ReLU(name="decoder_relu_2")(x)
x = layers.BatchNormalization(name="decoder_bn_2")(x)

x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", name="decoder_conv_transpose_layer_3")(x)
x = layers.ReLU(name="decoder_relu_3")(x)
x = layers.BatchNormalization(name="decoder_bn_3")(x)

x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", name="decoder_conv_transpose_layer_4")(x)
x = layers.ReLU(name="decoder_relu_4")(x)
x = layers.BatchNormalization(name="decoder_bn_4")(x)


x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same", name="decoder_conv_transpose_layer_5")(x)
'''x = layers.ReLU(name="decoder_relu_5")(x)
x = layers.BatchNormalization(name="decoder_bn_5")(x) '''

decoder_outputs = Activation("sigmoid", name="sigmoid_layer")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)))

# ---------------------------------------------------------------------------------------
# Define the VAE as a Model with a custom train_step
# ---------------------------------------------------------------------------------------

class VAE(keras.Model):
    def __init__(self, encoder, decoder, autoencoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.model = autoencoder
        '''self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")'''

    def summary(self):
        self.model.summary()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = 10 * reconstruction_loss\
                                                         + kl_loss
        train_loss(combined_loss)
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        train_reconstruction_loss(reconstruction_loss)
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.encoder.z_log_var - K.square(self.encoder.z_mean) -
                               K.exp(self.encoder.z_log_var), axis=1)
        train_kl_loss(kl_loss)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_kl_loss])

    def train(self, x_train, y_train, x_val, y_val, batch_size, num_epochs):

        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=False,
                       validation_data=(x_val, y_val))
# ---------------------------------------------------------------------------------------
# Train the VAE
# ---------------------------------------------------------------------------------------


vae = VAE(encoder, decoder, autoencoder)
vae.compile(LEARNING_RATE)
# vae.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
vae.train(x_train, x_train, x_val, x_val, BATCH_SIZE, EPOCHS)


