from VV_autoencoder import VAE
from functions import load_fsdd
import pathlib


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
path_features_matching_flute_val = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/NEW_HQ_FW_normalised_flute_VALID/"

x_train_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_train)
x_val_SPECTROGRAMS_PATH = pathlib.Path(path_features_matching_flute_val)

x_t = x_train_SPECTROGRAMS_PATH.iterdir()
x_v = x_val_SPECTROGRAMS_PATH.iterdir()

annotations = []
for file in x_v:
    name = file.name
    pitch = name[33:-16]
    print(pitch)
    annotations.append(pitch)
# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------

# vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)
vae = VAE.load("/nas/home/spol/Thesis/saved_model/REDUCTED/30-04-2022_00:28")


# ---------------------------------------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print('ollare tsne')
    # x_train = load_fsdd(x_train_SPECTROGRAMS_PATH)
    x_val = load_fsdd(x_val_SPECTROGRAMS_PATH)
    # print('x_train.shape: ', x_train.shape)
    print('x_val.shape: ', x_val.shape)
    # encoded_x_train = vae.tsne(x_train, perplexity=5, title='x_train')
    encoded_x_val = vae.tsne(x_val, perplexity=16, title='x_val', annotations=annotations)


print('debug')