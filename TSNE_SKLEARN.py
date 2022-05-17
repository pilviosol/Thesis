import numpy as np
from sklearn.manifold import TSNE
import numpy as np
from VV_autoencoder import VAE
from functions import load_fsdd
import pathlib
from CVAE_train import ones_val, zeros_val, cond01_val, cond10_val, cond_enc_val, cond_dec_val
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

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

perplexity = 6
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
    print('ollare tsne')

    x_val0 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val0 = [x_val0, ones_val, cond01_val]
    x_val1 = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val1 = [x_val1, zeros_val, cond10_val]

    encoded_inputs0 = vae.encoder.predict(x_val0)
    encoded_inputs1 = vae.encoder.predict(x_val1)

    encoded_inputs = np.concatenate((encoded_inputs0, encoded_inputs1), axis=0)
    labels = []
    for i in range(92):
        if i < 46:
            labels.append("red")
        else:
            labels.append("blue")
    labels = np.asarray(labels)

    X_embedded = TSNE(n_components=2,
                      perplexity=2,
                      learning_rate='auto',
                      init='random').fit_transform(encoded_inputs)


    def plot_2d(points, points_color, title):
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
        fig.suptitle(title, size=16)
        add_2d_scatter(ax, points, points_color)
        plt.show()


    def add_2d_scatter(ax, points, points_color, title=None):
        x, y = points.T
        ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())


    def plot_3d(points, points_color, title):
        x, y, z = points.T

        fig, ax = plt.subplots(
            figsize=(6, 6),
            facecolor="white",
            tight_layout=True,
            subplot_kw={"projection": "3d"},
        )
        fig.suptitle(title, size=16)
        col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
        ax.view_init(azim=-60, elev=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

        fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
        plt.show()


    plot_2d(X_embedded, labels, "TSNE-sklearn")
