import numpy as np
from VV_autoencoder import VAE
import pathlib
from functions import load_fsdd, interpolate
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------


spectrograms_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/inputs/"
spectrograms = pathlib.Path(spectrograms_path)
images_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/images/"
wav_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION/wav/"
SR = 16000


# ---------------------------------------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------------------------------------


vae = VAE.load("/nas/home/spol/Thesis/saved_model/REDUCTED/30-04-2022_14:54")


# ---------------------------------------------------------------------------------------------------------------------
# FEED THE MODEL, INTERPOLATE AND PREDICT INTERPOLATION
# ---------------------------------------------------------------------------------------------------------------------


spectrograms = load_fsdd(spectrograms)
encoded_spectrograms = vae.encoder.predict(spectrograms)
generated_image_vectors = np.asarray(interpolate(encoded_spectrograms[0].flatten(), encoded_spectrograms[1].flatten()))
generated_spectrograms = vae.decoder.predict(generated_image_vectors)


# ---------------------------------------------------------------------------------------------------------------------
# PLOT PREDICTIONS, SAVE IMAGES, RE-SYNTHESIZE AND SAVE AUDIO
# ---------------------------------------------------------------------------------------------------------------------


for idx, file in enumerate(generated_spectrograms):
    spectrogram = np.squeeze(file)
    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 64, 0, 512], aspect='auto')
    plt.colorbar()
    plt.title(idx)
    plt.show()
    plt.savefig(images_path + str(idx))
    plt.close()

    spectrogram = 10 ** (spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=32, hop_length=128)
    scipy.io.wavfile.write(wav_path + str(idx) + '.wav', SR, reconstructed)

print('debug')
