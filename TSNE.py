import numpy as np

from VV_autoencoder import VAE
from functions import load_fsdd
import pathlib
from CVAE_train import ones_val, zeros_val, cond01_val, cond10_val, cond_enc_val, cond_dec_val
from tsne import bh_sne
import matplotlib.pyplot as plt
"""
def bh_sne(
    data,
    pca_d=None,
    d=2,
    perplexity=30.0,
    theta=0.5,
    random_state=None,
    copy_data=False,
    verbose=False,
):
"""

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------

path_features_matching_flute_train = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/'
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)

x_t = x_train_SPECTROGRAMS_PATH.iterdir()

annotations0 = []
annotations1 = []

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
    annotations1.append(pitch + '_1')

annotations = [annotations0, annotations1]
annotations_new = annotations0 + annotations1

perplexity = 3
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


    for i, txt in enumerate(annotations[0]):
        plt.annotate(txt, (vis_x0[i], vis_y0[i]))

    for i, txt in enumerate(annotations[1]):
        plt.annotate(txt, (vis_x1[i], vis_y1[i]))

    plt.show()

    x_valA = load_fsdd(x_val_SPECTROGRAMS_PATH)
    x_valB = np.concatenate((x_valA, x_valA))
    x_val_new = [x_valB, cond_enc_val, cond_dec_val]
    encoded_inputs = vae.tsne(x_val_new, perplexity=perplexity, title='prova', annotations=annotations_new, color='green')




print('debug')
