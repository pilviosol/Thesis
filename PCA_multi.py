from sklearn.decomposition import PCA
import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
from functions import load_fsdd, plot_3d, plot_2d
import pathlib

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES, ANNOTATIONS
# ---------------------------------------------------------------------------------------------------------------------


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"


path_save_pca_images = "/nas/home/spol/Thesis/PCA/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)




# ENCODER CONDITIONING MATRICES

zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))
zeros_train = np.repeat(zeros, 708, axis=0)
zeros_val = np.repeat(zeros, 88, axis=0)

ones = np.ones([512, 64], dtype=float)
ones = np.expand_dims(ones, (-1, 0))
ones_train = np.repeat(ones, 708, axis=0)
ones_val = np.repeat(ones, 88, axis=0)

twos_train = np.add(ones_train, ones_train)
twos_val = np.add(ones_val, ones_val)

threes_train = np.add(ones_train, twos_train)
threes_val = np.add(ones_val, twos_val)


cond_enc_train = np.concatenate((ones_train, zeros_train, twos_train, threes_train), axis=0)
cond_enc_val = np.concatenate((ones_val, zeros_val, twos_val, threes_val), axis=0)


# DECODER CONDITIONING VECTORS
cond0001 = np.asarray([0, 0, 0, 1])
cond0001 = np.expand_dims(cond0001, axis=0)
cond0001_train = np.repeat(cond0001, 708, axis=0)
cond0001_val = np.repeat(cond0001, 88, axis=0)

cond0010 = np.asarray([0, 0, 1, 0])
cond0010 = np.expand_dims(cond0010, axis=0)
cond0010_train = np.repeat(cond0010, 708, axis=0)
cond0010_val = np.repeat(cond0010, 88, axis=0)

cond0100 = np.asarray([0, 1, 0, 0])
cond0100 = np.expand_dims(cond0100, axis=0)
cond0100_train = np.repeat(cond0100, 708, axis=0)
cond0100_val = np.repeat(cond0100, 88, axis=0)

cond1000 = np.asarray([1, 0, 0, 0])
cond1000 = np.expand_dims(cond1000, axis=0)
cond1000_train = np.repeat(cond1000, 708, axis=0)
cond1000_val = np.repeat(cond1000, 88, axis=0)


cond_dec_train = np.concatenate((cond0001_train, cond0010_train, cond0100_train, cond1000_train), axis=0)
cond_dec_val = np.concatenate((cond0001_val, cond0010_val, cond0100_val, cond1000_val), axis=0)



# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/18-05-2022_22:37")

# ---------------------------------------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare PCA_multi')

    # ------------------------------------
    # VALIDATION
    # ------------------------------------

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

    encoded_inputs_val_reducted = encoded_inputs_val[:, [-6, -5, -4, -3, -2, -1]]

    labels_val = []
    for i in range(352):
        if i < 88:
            labels_val.append("red")
        elif 88 <= i < 176:
            labels_val.append("blue")
        elif 176 <= i < 264:
            labels_val.append("green")
        else:
            labels_val.append("black")

    labels_val = np.asarray(labels_val)

    # ------------------------------------
    # TRAIN
    # ------------------------------------

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

    encoded_inputs_train_reducted = encoded_inputs_train[:, [-6, -5, -4, -3, -2, -1]]

    labels_train = []
    for i in range(2832):
        if i < 708:
            labels_train.append("red")
        elif 708 <= i < 1416:
            labels_train.append("blue")
        elif 1416 <= i < 2124:
            labels_train.append("green")
        else:
            labels_train.append("black")
    labels_train = np.asarray(labels_train)

    X_embedded_val = PCA(n_components=2, svd_solver='randomized').fit_transform(encoded_inputs_val_reducted)

    X_embedded_train = PCA(n_components=2, svd_solver='randomized').fit_transform(encoded_inputs_train_reducted)

    plot_2d(X_embedded_val, labels_val, "VAL_PCA_randomized_REDUCTED", path_save_pca_images)
    plot_2d(X_embedded_train, labels_train, "TRAIN_PCA_randomized_REDUCTED", path_save_pca_images)
    # plot_3d(X_embedded_val, labels_val, "VAL")
    # plot_3d(X_embedded_train, labels_train, "TRAIN")