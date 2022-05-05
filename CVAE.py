import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Concatenate
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

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
try:
    os.makedirs('/nas/home/spol/Thesis/saved_model/images/' + dt_string + '/')
except OSError:
    print("Creation of the directory  failed")

callback_list = []


