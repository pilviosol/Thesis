import numpy as np
from VV_autoencoder import VAE
import pathlib
from functions import load_fsdd, interpolate, denormalise
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
from WANDB import config


# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------


folder_number = str(9)
spectrogram_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/16052/inputs/input" + folder_number + "/"
spectrogram = pathlib.Path(spectrogram_path)
images_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/16052/images/image" + folder_number + "/"
wav_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/16052/wavs/wav" + folder_number + "/"
tsne_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/16052/TSNEs/"
min_max = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/string_folder_min_max.npy"

SR = config['sample_rate']
annotations = []
for i in range(10):
    annotations.append(str(i))


# ---------------------------------------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------------------------------------


vae = VAE.load("/nas/home/spol/Thesis/saved_model/CVAE/16-05-2022_22:54")


# ---------------------------------------------------------------------------------------------------------------------
# FEED THE MODEL, INTERPOLATE AND PREDICT INTERPOLATION
# ---------------------------------------------------------------------------------------------------------------------


spectrogram = load_fsdd(spectrogram)
spectrograms = np.concatenate((spectrogram, spectrogram), axis=0)


ones = np.ones([512, 64], dtype=float)
ones = np.expand_dims(ones, (-1, 0))

zeros = np.zeros([512, 64], dtype=float)
zeros = np.expand_dims(zeros, (-1, 0))

cond_enc = np.concatenate((ones, zeros), axis=0)

cond01 = np.asarray([0, 1])
cond01 = np.expand_dims(cond01, axis=0)

cond10 = np.asarray([1, 0])
cond10 = np.expand_dims(cond10, axis=0)

cond_dec = np.concatenate((cond01, cond10), axis=0)

spectrograms_full = [spectrograms, cond_enc, cond_dec]

encoded_spectrograms = vae.encoder.predict(spectrograms_full)
generated_image_vectors = np.asarray(interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[1].flatten()))
generated_spectrograms = vae.decoder.predict(generated_image_vectors)


# ---------------------------------------------------------------------------------------------------------------------
# TSNE ON GENERATED POINTS
# ---------------------------------------------------------------------------------------------------------------------


encoded_generated = vae.tsne_interpolation(generated_image_vectors, perplexity=3, title='x_val',
                                           annotations=annotations, color='red',
                                           save_image_path=tsne_path + folder_number)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT PREDICTIONS, SAVE IMAGES, RE-SYNTHESIZE AND SAVE AUDIO
# ---------------------------------------------------------------------------------------------------------------------


min_and_max = np.load(min_max)
minimum = min_and_max[0][1]
maximum = min_and_max[0][0]


for idx, file in enumerate(generated_spectrograms):
    spectrogram = np.squeeze(file)

    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.title(idx)
    plt.savefig(images_path + str(idx))
    plt.show()
    plt.close()
    denormalised_spectrogram = denormalise(spectrogram, minimum, maximum)
    spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
    scipy.io.wavfile.write(wav_path + str(idx) + '.wav', SR, reconstructed)


print('debug')
