import numpy as np
from sklearn.manifold import TSNE
import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
from functions import load_fsdd
import pathlib
from CVAE_train_multi import ones_val, zeros_val, twos_val, threes_val, cond0001_val, cond0010_val, cond0100_val, \
    cond1000_val, \
    ones_train, zeros_train, twos_train, threes_train, cond0001_train, cond0010_train, cond0100_train, cond1000_train
import matplotlib.pyplot as plt
from matplotlib import ticker


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

annotations = [annotations0, annotations1]
annotations_new = annotations0 + annotations1

perplexity = 6
colors = ['red', 'blue']

# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/18-05-2022_22:37")

# ---------------------------------------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare tsne_multi')

    x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val0 = [x_val, zeros_val, cond0001_val]
    x_val1 = [x_val, ones_val, cond0010_val]
    x_val2 = [x_val, twos_val, cond0100_val]
    x_val3 = [x_val, threes_val, cond1000_val]

    encoded_inputs_val0 = vae.encoder.predict(x_val0)
    encoded_inputs_val1 = vae.encoder.predict(x_val1)
    encoded_inputs_val2 = vae.encoder.predict(x_val2)
    encoded_inputs_val3 = vae.encoder.predict(x_val3)

    encoded_inputs_val = np.concatenate(
        (encoded_inputs_val0, encoded_inputs_val1, encoded_inputs_val2, encoded_inputs_val3), axis=0)
    labels_val = []
    for i in range(184):
        if i < 46:
            labels_val.append("red")
        elif 46 <= i < 92:
            labels_val.append("blue")
        elif 92 <= i < 138:
            labels_val.append("green")
        else:
            labels_val.append("black")
                     
    labels_val = np.asarray(labels_val)

    X_embedded_val = TSNE(n_components=3,
                          perplexity=5,
                          learning_rate='auto',
                          init='random').fit_transform(encoded_inputs_val)

    x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_train0 = [x_train, ones_train, cond0001_train]
    x_train1 = [x_train, zeros_train, cond0010_train]
    x_train2 = [x_train, twos_train, cond0100_train]
    x_train3 = [x_train, threes_train, cond1000_train]

    encoded_inputs_train0 = vae.encoder.predict(x_train0)
    encoded_inputs_train1 = vae.encoder.predict(x_train1)
    encoded_inputs_train2 = vae.encoder.predict(x_train2)
    encoded_inputs_train3 = vae.encoder.predict(x_train3)

    encoded_inputs_train = np.concatenate(
        (encoded_inputs_train0, encoded_inputs_train1, encoded_inputs_train2, encoded_inputs_train3), axis=0)
    
    labels_train = []
    
    for i in range(3296):
        if i < 824:
            labels_train.append("red")
        elif 824 <= i < 1648:
            labels_train.append("blue")
        elif 1648 <= i < 2472:
            labels_train.append("green")
        else:
            labels_train.append("black")
    labels_train = np.asarray(labels_train)

    X_embedded_train = TSNE(n_components=3,
                            perplexity=2,
                            learning_rate='auto',
                            init='random').fit_transform(encoded_inputs_train)

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
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

        # fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
        plt.show()


    # plot_2d(X_embedded_val, labels_val, "VAL")
    # plot_2d(X_embedded_train, labels_train, "TRAIN")
    plot_3d(X_embedded_val, labels_val, "VAL")
    plot_3d(X_embedded_train, labels_train, "TRAIN")
