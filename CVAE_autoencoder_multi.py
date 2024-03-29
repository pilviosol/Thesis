import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from WANDB import config
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
import matplotlib.pyplot as plt
from datetime import datetime
from tsne import bh_sne

# ---------------------------------------------------------------------------------------------------------------------
# DATE, LOSSES, EARLY STOPPING
# ---------------------------------------------------------------------------------------------------------------------


tf.compat.v1.disable_eager_execution()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_kl_loss = tf.keras.metrics.Mean(name="train_kl_loss")
train_reconstruction_loss = tf.keras.metrics.Mean(name="train_reconstruction_loss")

earlystopping = EarlyStopping(monitor='_calculate_reconstruction_loss', patience=50, verbose=1, restore_best_weights=True)


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
try:
    os.makedirs('/nas/home/spol/Thesis/saved_model/images/' + dt_string + '/')
except OSError:
    print("Creation of the directory  failed")

callback_list = []


# ---------------------------------------------------------------------------------------------------------------------
# CREATING CLASS VAE
# ---------------------------------------------------------------------------------------------------------------------


class CVAEMulti:
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
        self.input_shape = input_shape  # [512, 256, 1]
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
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

    def train(self, x_train, y_train, x_val, y_val, batch_size, num_epochs):
        callback_list.append(WandbCallback())

        def plot_and_save_while_training(epoch, logs):

            if epoch % 20 == 0:
                a = self.model.predict(x_val)

                for i in range(len(x_val[0])):
                    element = a[i]
                    element = np.squeeze(element)

                    if i % 5 == 0:
                        fig = plt.figure()
                        img = plt.imshow(element, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                                         aspect='auto')
                        title = 'epoch_' + str(epoch) + '_spectrogram_' + str(i)
                        plt.title(title)
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig('/nas/home/spol/Thesis/saved_model/images/' + dt_string + '/' + title)
                        plt.close()
                        wandb.log({"Validation set plots": [wandb.Image(fig, caption=title)]})

        callback_list.append(LambdaCallback(on_epoch_end=plot_and_save_while_training))
        callback_list.append(earlystopping)

        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       callbacks=callback_list,
                       validation_data=(x_val, y_val))

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
        autoencoder = CVAEMulti(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = reconstruction_loss + self.kl_loss_weight * kl_loss
        train_loss(combined_loss)
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = np.prod((512, 256)) * K.mean(K.square(error), axis=[1, 2, 3])
        train_reconstruction_loss(reconstruction_loss)
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=-1)
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
        # decoder_output2 = self._add_decoder_output2(decoder_output)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        X = Input(shape=self.latent_space_dim + 4, name="decoder_input")
        return X

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
        x = BatchNormalization(axis=-1, name=f"decoder_bn_{layer_num}")(x)
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

    def _add_decoder_output2(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=1,
            padding="same",
            name=f"decoder2_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        encoder_cond = self._add_encoder_cond()
        decoder_cond = self._add_decoder_cond()
        encoder_concat = self._add_encoder_concatenation(encoder_input, encoder_cond)
        conv_layers = self._add_conv_layers(encoder_concat)
        bottleneck = self._add_bottleneck(conv_layers)
        bottleneck_concateated = self._add_bottleneck_concatenation(bottleneck, decoder_cond)
        self._model_input = [encoder_input, encoder_cond, decoder_cond]
        self.encoder = Model([encoder_input, encoder_cond, decoder_cond], bottleneck_concateated, name="encoder")

    '''def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")'''

    def _add_encoder_input(self):
        X = Input(shape=self.input_shape, name="encoder_input")
        return X

    def _add_encoder_cond(self):
        cond_enc = Input(shape=self.input_shape, name="encoder_cond_input")
        return cond_enc

    def _add_encoder_concatenation(self, x, cond_enc):
        inputs = Concatenate(axis=-1, name="encoder_concat_input")([x, cond_enc])
        return inputs

    def _add_decoder_cond(self):
        cond_dec = Input(shape=4, name="decoder_cond_input")
        return cond_dec

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
        x = BatchNormalization(axis=-1, name=f"encoder_bn_{layer_number}")(x)
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

    def _add_bottleneck_concatenation(self, z, cond_dec):
        x = Concatenate(axis=-1, name="bottleneack_concat")([z, cond_dec])
        return x

    def tsne(self, x_train, perplexity, title, annotations, color):

        encoded_inputs = self.encoder.predict(x_train)

        # perform t-SNE embedding
        vis_data = bh_sne(encoded_inputs.astype('float64'), perplexity=perplexity)

        # plot the result
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), s=100, c=color)
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title(title)
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (vis_x[i], vis_y[i]))
        plt.show()
        return encoded_inputs

    def tsne_nuova(self, x_val0, x_val1, perplexity, title, annotations, colors):

        encoded_inputs0 = self.encoder.predict(x_val0)
        encoded_inputs1 = self.encoder.predict(x_val1)

        # perform t-SNE embedding
        vis_data0 = bh_sne(encoded_inputs0.astype('float64'), perplexity=perplexity)
        vis_data1 = bh_sne(encoded_inputs1.astype('float64'), perplexity=perplexity)

        # plot the result
        vis_x0 = vis_data0[:, 0]
        vis_y0 = vis_data0[:, 1]

        vis_x1 = vis_data1[:, 0]
        vis_y1 = vis_data1[:, 1]

        fig, plot = plt.subplots(1, 1)

        plot.scatter(vis_x0, vis_y0, s=100, c=colors[0])
        plot.scatter(vis_x1, vis_y1, s=100, c=colors[1])
        fig.title(title)
        '''
        for i, txt in enumerate(annotations[0]):
            plt.annotate(txt, (vis_x0[i], vis_y0[i]))

        for i, txt in enumerate(annotations[1]):
            plt.annotate(txt, (vis_x1[i], vis_y1[i])) '''

        fig.savefig('/nas/home/spol/Thesis/TSNE/')
        fig.show()
        return encoded_inputs0, encoded_inputs1

    def tsne_hq(self, x_train, interpolation_points, perplexity, title, annotations, color):

        encoded_inputs = self.encoder.predict(x_train)

        # perform t-SNE embedding
        vis_data = bh_sne(encoded_inputs.astype('float64'), perplexity=perplexity)
        interp_data = bh_sne(interpolation_points.astype('float64'), perplexity=perplexity)

        # plot the result
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), s=1000, c=color[0])
        plt.scatter(vis_x, interp_data, cmap=plt.cm.get_cmap("jet", 10), s=1000, c=color[1])

        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title(title)
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (vis_x[i], vis_y[i]))
        plt.show()
        return encoded_inputs

    def tsne_interpolation(self, interpolation_points, perplexity, title, annotations, color, save_image_path):

        # perform t-SNE embedding
        vis_data = bh_sne(interpolation_points.astype('float64'), perplexity=perplexity)

        # plot the result
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), s=100, c=color[0])

        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title(title)
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (vis_x[i], vis_y[i]))
        plt.savefig(save_image_path)
        plt.show()
        return vis_data


if __name__ == "__main__":
    autoencoder = CVAEMulti(
        input_shape=(512, 64, 1),
        conv_filters=(16*4, 32*4, 64*4, 128*4, 256*4),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=64
    )
    autoencoder.summary()
