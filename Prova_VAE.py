import wandb

#from __future__ import absolute_import, division, print_function, unicode_literals
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
    "epochs": 2000,
    "learning_rate": 0.0001,
    "batch_size": 64,
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
        train_vn_stft.append(feature_reshaped)

print('porco')

'''
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
'''
#history = vae.fit(train_vn_stft, train_vn_stft, epochs=config['epochs'], batch_size=config['batch_size'])