import numpy as np
from VV_autoencoder import VAE
import pathlib
from functions import load_fsdd, interpolate, denormalise
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------

folder_number = str(9)
spectrograms_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/inputs/input" + folder_number + "/"
spectrograms = pathlib.Path(spectrograms_path)
images_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/images/image" + folder_number + "/"
wav_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/wavs/wav" + folder_number + "/"
min_max = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/string_folder_min_max.npy"
SR = 16000
annotations = []
for i in range(10):
    annotations.append(str(i))


# ---------------------------------------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------------------------------------


vae = VAE.load("/nas/home/spol/Thesis/saved_model/REDUCTED/30-04-2022_00:28")


# ---------------------------------------------------------------------------------------------------------------------
# FEED THE MODEL, INTERPOLATE AND PREDICT INTERPOLATION
# ---------------------------------------------------------------------------------------------------------------------


spectrograms = load_fsdd(spectrograms)
encoded_spectrograms = vae.encoder.predict(spectrograms)
generated_image_vectors = np.asarray(interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[1].flatten()))
generated_spectrograms = vae.decoder.predict(generated_image_vectors)


# ---------------------------------------------------------------------------------------------------------------------
# TSNE ON GENERATED POINTS
# ---------------------------------------------------------------------------------------------------------------------


encoded_generated = vae.tsne_interpolation(generated_image_vectors, perplexity=3, title='x_val', annotations=annotations, color='red')


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
    plt.title(idx)
    plt.savefig(images_path + str(idx))
    plt.show()
    plt.close()
    denormalised_spectrogram = denormalise(spectrogram, minimum, maximum)
    spectrogram = 10 ** (denormalised_spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
    scipy.io.wavfile.write(wav_path + str(idx) + '.wav', SR, reconstructed)



print('debug')