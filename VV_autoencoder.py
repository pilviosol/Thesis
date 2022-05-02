import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from WANDB import config
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LambdaCallback
import matplotlib.pyplot as plt
from datetime import datetime
from tsne import bh_sne

tf.compat.v1.disable_eager_execution()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_kl_loss = tf.keras.metrics.Mean(name="train_kl_loss")
train_reconstruction_loss = tf.keras.metrics.Mean(name="train_reconstruction_loss")

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=1, restore_best_weights=False)

# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
try:
    os.makedirs('/nas/home/spol/Thesis/saved_model/images/' + dt_string + '/')
except OSError:
    print("Creation of the directory  failed")

callback_list = []


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    # with strategy.scope():
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        # self.reconstruction_loss_weight = 1000000
        # self.reconstruction_loss_weight = config['kl_alpha']
        self.kl_loss_weight = config['kl_beta']

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_kl_loss])

    def train_overfit(self, x_train, y_train, batch_size, num_epochs):
        callback_list.append(WandbCallback())

        def plot_and_save_while_training_overfit(epoch, logs):

            if epoch % 10 == 0:
                a = self.model.predict(y_train)

                for i in range(len(y_train)):
                    element = a[i]
                    element = np.squeeze(element)

                    fig = plt.figure()
                    img = plt.imshow(element, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                                     aspect='auto')
                    title = str(epoch) + '_' + str(i)
                    plt.title(title)
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig('/nas/home/spol/Thesis/saved_model/images_overfit/' + title)
                    plt.close()
                    wandb.log({"Y_train plots": [wandb.Image(fig, caption=title)]})
        callback_list.append(LambdaCallback(on_epoch_end=plot_and_save_while_training_overfit))

        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=False,
                       callbacks=callback_list)

    def train(self, x_train, y_train, x_val, y_val, batch_size, num_epochs):
        callback_list.append(WandbCallback())

        def plot_and_save_while_training(epoch, logs):

            if epoch % 5 == 0:
                a = self.model.predict(y_val)

                for i in range(len(y_val)):
                    element = a[i]
                    element = np.squeeze(element)

                    if i % 10 == 0:
                        fig = plt.figure()
                        img = plt.imshow(element, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                                         aspect='auto')
                        title = str(epoch) + '_' + str(i)
                        plt.title(title)
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig('/nas/home/spol/Thesis/saved_model/images/' + dt_string + '/' + title)

                        plt.close()
                        wandb.log({"Validation set plots": [wandb.Image(fig, caption=title)]})

        callback_list.append(LambdaCallback(on_epoch_end=plot_and_save_while_training))
        callback_list.append(callback)
        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=False,
                       callbacks=callback_list,
                       validation_data=(x_val, y_val))

        '''
        t_loss = train_loss.result()
        wandb.log({"train_loss": t_loss.numpy(), "global_step": num_epochs})
        t_kl_loss = train_kl_loss.result()
        wandb.log({"train_kl_loss": t_kl_loss.numpy(), "global_step": num_epochs})
        t_reconstruction_loss = train_reconstruction_loss.result()
        wandb.log({"reconstruction_loss": t_reconstruction_loss.numpy(), "global_step": num_epochs})
        '''

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
        combined_loss = reconstruction_loss + self.kl_loss_weight*kl_loss
        train_loss(combined_loss)
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        train_reconstruction_loss(reconstruction_loss)
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
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

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="encoder_output")([self.mu, self.log_variance])
        return x

    def tsne(self, x_train, perplexity, title, annotations):

        encoded_inputs = self.encoder.predict(x_train)

        # perform t-SNE embedding
        vis_data = bh_sne(encoded_inputs.astype('float64'), perplexity=perplexity)

        # plot the result
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), s=1)
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title(title)
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (vis_x[i], vis_y[i]))
        plt.show()
        return encoded_inputs




if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
