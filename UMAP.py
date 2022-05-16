import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from VV_autoencoder import VAE
import pathlib
import pandas as pd
import umap
from functions import load_fsdd
from CVAE_train import ones_val, zeros_val, cond01_val, cond10_val, cond_enc_val, cond_dec_val

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES, ANNOTATIONS
# ---------------------------------------------------------------------------------------------------------------------


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)

x_t = x_train_SPECTROGRAMS_PATH.iterdir()

annotations0 = []
annotations1 = []
'''
x_v = x_val_SPECTROGRAMS_PATH.iterdir()
for file in x_v:
    name = file.name
    pitch = name[33:-16]
    print(pitch)
    annotations0.append(pitch + '_0')

x_v = x_val_SPECTROGRAMS_PATH.iterdir()
for file in x_v:
    name = file.name
    pitch = name[33:-16]
    print(pitch)
    annotations1.append(pitch + '_1') '''

annotations = [annotations0, annotations1]
annotations_new = annotations0 + annotations1

colors = ['red', 'blue']


# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = VAE.load("/nas/home/spol/Thesis/saved_model/CVAE/11-05-2022_12:30")


# ---------------------------------------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare umap')

    x_val0 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val0 = [x_val0, ones_val, cond01_val]
    x_val1 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val1 = [x_val1, zeros_val, cond10_val]

    encoded_inputs0 = vae.encoder.predict(x_val0)
    encoded_inputs1 = vae.encoder.predict(x_val1)
    reducer = umap.UMAP(random_state=42)
    reducer.fit(encoded_inputs0)
    reducer.fit(encoded_inputs1)

    embedding = reducer.transform(encoded_inputs0)

    plt.scatter(embedding[:, 0], embedding[:, 1], c='red', cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24)
    plt.show()