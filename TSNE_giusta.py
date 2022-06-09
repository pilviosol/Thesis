from sklearn.manifold import TSNE
import numpy as np
from CVAE_autoencoder_multi import CVAEMulti
from functions import load_fsdd, plot_3d, plot_2d
import pathlib
from CVAE_train_multi import ones_val, zeros_val, twos_val, threes_val, cond0001_val, cond0010_val, cond0100_val, \
    cond1000_val, \
    ones_train, zeros_train, twos_train, threes_train, cond0001_train, cond0010_train, cond0100_train, cond1000_train
import tensorflow.keras


# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES, ANNOTATIONS
# ---------------------------------------------------------------------------------------------------------------------


path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"

path_save_tsne_images = "/nas/home/spol/Thesis/TSNE_GIUSTA/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)

annotations0 = []
annotations1 = []

annotations = [annotations0, annotations1]
annotations_new = annotations0 + annotations1

perplexity_val = 30
perplexity_train = 30

# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------


# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = CVAEMulti.load("/nas/home/spol/Thesis/saved_model/CVAE_multi/18-05-2022_22:37")


mu_layer_name = 'mu'
mu_layer_model = tensorflow.keras.Model(inputs=vae._model_input, outputs=vae.encoder.get_layer(mu_layer_name).output)

lv_layer_name = 'log_variance'
lv_layer_model = tensorflow.keras.Model(inputs=vae._model_input, outputs=vae.encoder.get_layer(lv_layer_name).output)


# ---------------------------------------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('ollare tsne_multi')

    # ------------------------------------
    # VALIDATION
    # ------------------------------------

    x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_val0 = [x_val, zeros_val, cond0001_val]
    x_val1 = [x_val, ones_val, cond0010_val]
    x_val2 = [x_val, twos_val, cond0100_val]
    x_val3 = [x_val, threes_val, cond1000_val]

    encoded_inputs_val0 = mu_layer_model.predict(x_val0)
    encoded_inputs_val1 = mu_layer_model.predict(x_val1)
    encoded_inputs_val2 = mu_layer_model.predict(x_val2)
    encoded_inputs_val3 = mu_layer_model.predict(x_val3)

    encoded_inputs_val = np.concatenate(
        (encoded_inputs_val0, encoded_inputs_val1, encoded_inputs_val2, encoded_inputs_val3), axis=0)

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

    encoded_inputs_train0 = mu_layer_model.predict(x_train0)
    encoded_inputs_train1 = mu_layer_model.predict(x_train1)
    encoded_inputs_train2 = mu_layer_model.predict(x_train2)
    encoded_inputs_train3 = mu_layer_model.predict(x_train3)

    encoded_inputs_train = np.concatenate(
        (encoded_inputs_train0, encoded_inputs_train1, encoded_inputs_train2, encoded_inputs_train3), axis=0)

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

    for perplexity in range(20):
        print(perplexity)
        if perplexity > 2:
            X_embedded_val = TSNE(n_components=2,
                                  perplexity=perplexity,
                                  early_exaggeration=12,
                                  learning_rate='auto',
                                  init='random').fit_transform(encoded_inputs_val)

            X_embedded_train = TSNE(n_components=2,
                                    perplexity=perplexity,
                                    early_exaggeration=12,
                                    learning_rate='auto',
                                    init='random').fit_transform(encoded_inputs_train)

            plot_2d(X_embedded_val, labels_val, "VAL_perplexity_" + str(perplexity), path_save_tsne_images)
            plot_2d(X_embedded_train, labels_train, "TRAIN_perplexity_" + str(perplexity), path_save_tsne_images)
            # plot_3d(X_embedded_val, labels_val, "VAL")
            # plot_3d(X_embedded_train, labels_train, "TRAIN")
