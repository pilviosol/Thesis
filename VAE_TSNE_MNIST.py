import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
# from tensorflow.keras import objectives
from matplotlib import pyplot as plt
from tsne import bh_sne
import tensorflow as tf


image_size = 28
original_dim = image_size * image_size
latent_dim = 32
intermediate_dim = 128
n_samples = 30000


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train = x_train[:n_samples]
y_train = y_train[:n_samples]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=1.)
    sampled_point = z_mean + K.exp(z_log_std / 2) * epsilon
    return sampled_point


def vae_loss(x, x_decoded_mean):
    xent_loss = K.mean(K.sum(K.square(x - x_decoded_mean), axis=-1), axis=-1)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return xent_loss + kl_loss


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name="1")(x)
h = Dense(intermediate_dim, activation='relu', name="2")(h)
z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_std = Dense(latent_dim, name="z_log_std")(h)
z = Lambda(sampling, name="Lambda")([z_mean, z_log_std])
decoder_h1 = Dense(intermediate_dim, activation='relu', name="dioporco")
decoder_h2 = Dense(intermediate_dim, activation='relu', name="diocane")
decoder_mean = Dense(original_dim, activation='sigmoid', name="diomerda")
h1_decoded = decoder_h1(z)
h2_decoded = decoder_h2(h1_decoded)
x_decoded_mean = decoder_mean(h2_decoded)


vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss, experimental_run_tf_function=False)


vae.fit(x_train, x_train,
        shuffle=True,
        epochs=50,
        batch_size=256,
        validation_data=(x_test, x_test))


encoded_inputs = encoder.predict(x_train)


# perform t-SNE embedding
vis_data = bh_sne(encoded_inputs.astype('float64'))


# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_train, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()





