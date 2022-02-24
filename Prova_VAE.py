import wandb

# from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
from utils import *

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

import random
import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import configparser
import random
import json
import matplotlib.pyplot as plt

set_gpu(0)

wandb.init(project="my-test-project", entity="pilviosol")

wandb.config = {
    "sample_rate": 44100,
    "hop_length": 128,
    "bins_per_octave": 48,
    "num_octaves": 8,
    "n_iter": 1,
    "latent_dim": 256,
    "n_units": 2048,
    "kl_beta": 0.0005,
    "batch_norm": True,
    "output_activation": "relu",
    "epochs": 500,
    "learning_rate": 0.0001,
    "batch_size": 384,
    "buffer_size": 60000,
    "buffer_size_dataset": True,
    "continue_training": False,
    "max_ckpts_to_keep": 2,
    "checkpoint_epochs": 15,
    "save_best_only": True,
    "learning_schedule": False,
    "early_patience_epoch": 50,
    "early_delta": 1e-9,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999
}
config = wandb.config

train_vn = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train_256')
train_vn_dir = train_vn.iterdir()
train_vn_stft = []

for idx, feature in enumerate(train_vn_dir):
    feature_name = feature.name
    if "CQT." in feature_name:
        feature_np = np.load(feature)
        feature_reshaped = feature_np[0:1024, 0:256]
        print("feature.name: ", feature_name)
        print("feature.shape: ", feature_reshaped.shape)
        train_vn_stft.append(np.transpose(feature_reshaped))

print(train_vn_stft[0].shape)


# Define Sampling Layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Train

if not config['continue_training']:
    # Define encoder model.
    n_bins = int(config['num_octaves'] * config['bins_per_octave'])

    original_dim = n_bins
    original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(config['n_units'], activation='relu')(original_inputs)
    z_mean = layers.Dense(config['latent_dim'], name='z_mean')(x)
    z_log_var = layers.Dense(config['latent_dim'], name='z_log_var')(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')
    encoder.summary()

    # Define decoder model.
    latent_inputs = tf.keras.Input(shape=(config['latent_dim'],), name='z_sampling')
    x = layers.Dense(config['n_units'], activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation=config['output_activation'])(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    decoder.summary()

    outputs = decoder(z)
    # Define VAE model.
    vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')
    vae.summary()

    # Add KL divergence regularization loss.
    kl_loss = - config['kl_beta'] * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)

    if config['learning_schedule']:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            config['learning_rate'] * 100,
            decay_steps=int(config['epochs'] * 0.8),
            decay_rate=0.96,
            staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=config['adam_beta_1'],
                                         beta_2=config['adam_beta_2'])

    vae.compile(optimizer,
                loss=tf.keras.losses.MeanSquaredError())

history = vae.fit(train_vn_stft, train_vn_stft, epochs=config['epochs'], batch_size=config['batch_size'])

'''
# Generate examples
print("Generating examples...")
my_examples_folder = workdir.joinpath('audio_examples')
for f in os.listdir(my_audio):
    print("Examples for {}".format(os.path.splitext(f)[0]))
    file_path = my_audio.joinpath(f)
    my_file_duration = librosa.get_duration(filename=file_path)
    my_offset = random.randint(0, int(my_file_duration) - example_length)
    s, fs = librosa.load(file_path, duration=example_length, offset=my_offset, sr=None)
    # Get the CQT magnitude
    print("Calculating CQT")
    C_complex = librosa.cqt(y=s, sr=fs, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
    C = np.abs(C_complex)
    # Invert using Griffin-Lim
    # y_inv = librosa.griffinlim_cqt(C, sr=fs, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave)
    # And invert without estimating phase
    # y_icqt = librosa.icqt(C, sr=fs, hop_length=hop_length, bins_per_octave=bins_per_octave)
    # y_icqt_full = librosa.icqt(C_complex, hop_length=hop_length, sr=fs, bins_per_octave=bins_per_octave)
    C_32 = C.astype('float32')
    y_inv_32 = librosa.griffinlim_cqt(C, sr=fs, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave,
                                      dtype=np.float32)
    ## Generate the same CQT using the model
    my_array = np.transpose(C_32)
    test_dataset = tf.data.Dataset.from_tensor_slices(my_array).batch(batch_size).prefetch(AUTOTUNE)
    output = tf.constant(0., dtype='float32', shape=(1, n_bins))

    print("Working on regenerating cqt magnitudes with the DL model")
    for step, x_batch_train in enumerate(test_dataset):
        reconstructed = vae(x_batch_train)
        output = tf.concat([output, reconstructed], 0)

    output_np = np.transpose(output.numpy())
    output_inv_32 = librosa.griffinlim_cqt(output_np[1:],
                                           sr=fs, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave,
                                           dtype=np.float32)

    if normalize_examples:
        output_inv_32 = librosa.util.normalize(output_inv_32)
    print("Saving audio files...")
    my_audio_out_fold = my_examples_folder.joinpath(os.path.splitext(f)[0])
    os.makedirs(my_audio_out_fold, exist_ok=True)
    librosa.output.write_wav(my_audio_out_fold.joinpath('original.wav'),
                             s, sample_rate)
    librosa.output.write_wav(my_audio_out_fold.joinpath('original-icqt+gL.wav'),
                             y_inv_32, sample_rate)
    librosa.output.write_wav(my_audio_out_fold.joinpath('VAE-output+gL.wav'),
                             output_inv_32, sample_rate)

# Generate a plot for loss
print("Generating loss plot...")
history_dict = history.history
fig = plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0., 0.01)
plt.plot(history_dict['loss'])
fig.savefig(workdir.joinpath('my_history_plot.pdf'), dpi=300)

print('bye...')
'''

