from VV_autoencoder import VAE
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa.display
import librosa
from functions import feature_calculation, normalise_set_and_save_min_max, denormalise
from WANDB import config
import scipy.io.wavfile

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------

flute_audio_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute/flute_acoustic_002-091-050.wav"
vocal_audio_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal/vocal_synthetic_003-091-050.wav"
flute_folder_audio_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute/"
vocal_folder_audio_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal/"
flute_features = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute_features/"
vocal_features = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal_features/"
flute_features_normalised = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/flute_features_normalised/"
vocal_features_normalised = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal_features_normalised/"

generated_vocal_normalised = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/generated_vocal_features_normalised/"
generated_vocal_denormalised = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/generated_vocal_features_denormalised/"
generated_vocal_audio = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal_generated/"
generated_vocal_features = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_OVERFIT_SUBSET/vocal_generated_features/"

SR = config['sample_rate']

# ---------------------------------------------------------------------------------------------------------------------
# PLOT AUDIO WAVEFORM OF INPUT AND OUPUT
# ---------------------------------------------------------------------------------------------------------------------


flute, _ = librosa.load(flute_audio_path, res_type='kaiser_fast', mono=True, sr=None)
vocal, _ = librosa.load(vocal_audio_path, res_type='kaiser_fast', mono=True, sr=None)

fig, ax = plt.subplots(nrows=2)

librosa.display.waveshow(flute, sr=SR, ax=ax[0])
ax[0].set(title='flute')
ax[0].label_outer()

librosa.display.waveshow(vocal, sr=SR, ax=ax[1])
ax[1].set(title='vocal')
ax[1].label_outer()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------


feature_calculation(flute_folder_audio_path, flute_features)
feature_calculation(vocal_folder_audio_path, vocal_features)

# ---------------------------------------------------------------------------------------------------------------------
# NORMALISE SPECTROGRAMS AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------


min_max_flute = normalise_set_and_save_min_max(flute_features, flute_features_normalised)
min_max_vocal = normalise_set_and_save_min_max(vocal_features, vocal_features_normalised)

# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL AND DEFINE GENERATE FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


vae = VAE.load("/nas/home/spol/Thesis/saved_model/OVERFIT")


def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


# ---------------------------------------------------------------------------------------------------------------------
# GENERATE THE SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------


normalised_features_path = pathlib.Path(flute_features_normalised).iterdir()

for file in sorted(normalised_features_path):
    name = file.name
    spectrogram = np.load(file)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print('spectrogram expanded dims: ', spectrogram.shape)

    generated_spectrogram, latent_representation = generate(spectrogram)
    print('generated_spectrogram.shape:', generated_spectrogram.shape)
    np.save(generated_vocal_normalised + 'GENERATED_' + name, generated_spectrogram)

# ---------------------------------------------------------------------------------------------------------------------
# PLOT INPUT, OUTPUT and EXPECTED OUTPUT OF THE NETWORK
# ---------------------------------------------------------------------------------------------------------------------


VAE_input = pathlib.Path(flute_features_normalised).iterdir()
VAE_output = pathlib.Path(generated_vocal_normalised).iterdir()
VAE_expected_output = pathlib.Path(vocal_features_normalised).iterdir()

for inpt, output, expected_output in zip(sorted(VAE_input), sorted(VAE_output), sorted(VAE_expected_output)):
    input_name = inpt.name
    print('input_name: ', input_name)
    input_spectrogram = np.load(inpt)

    output_name = output.name
    print('output_name: ', output_name)
    output_spectrogram = np.load(output)
    out_spectrogram = np.squeeze(output_spectrogram)

    expected_output_name = expected_output.name
    expected_output_spectrogram = np.load(expected_output)
    print('expected_output_spectrogram dims: ', expected_output_spectrogram.shape)

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # VAE INPUT
    img1 = ax1.imshow(input_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax1.set_title('INPUT')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    # VAE OUTPUT
    img2 = ax2.imshow(out_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax2.set_title('OUTPUT')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)

    # EXPECTED OUTPUT
    img3 = ax3.imshow(expected_output_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax3.set_title('EXPECTED OUTPUT')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(img3, cax=cax3)

    plt.tight_layout()
    plt.show()
    plt.close()

    print("MSE(net out spec, expected out spec): ", np.mean((out_spectrogram - expected_output_spectrogram) ** 2))
    print("NMSE(net out spec, expected out spec): ", np.mean((out_spectrogram - expected_output_spectrogram) ** 2 /
                                                             expected_output_spectrogram ** 2))

# ---------------------------------------------------------------------------------------------------------------------
# DENORMALISE THE GENERATED SPECTROGRAM
# ---------------------------------------------------------------------------------------------------------------------


generated_spectrogram_path = pathlib.Path(generated_vocal_normalised).iterdir()

for idx, file in enumerate(sorted(generated_spectrogram_path)):
    print('idx: ', idx)
    name = file.name
    gen_spectrogram = np.load(file)
    gen_spectrogram = np.squeeze(gen_spectrogram)
    # denormalised_spectrogram = denormalise_given_min_max(gen_spectrogram, min_max_values[0][0], min_max_values[0][1])
    denormalised_spectrogram = denormalise(gen_spectrogram, min_max_flute[idx][0], min_max_flute[idx][1])
    np.save(generated_vocal_denormalised + name, denormalised_spectrogram)

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE DENORMALISED GENERATED SPECTROGRAM AND RESYNTHESIZE IT
# ---------------------------------------------------------------------------------------------------------------------


denormalised_generated_spectrogram_path = pathlib.Path(generated_vocal_denormalised).iterdir()
for file in sorted(denormalised_generated_spectrogram_path):
    name = file.name
    name = name[0:-4]
    print(name)

    denorm_spectrogram = np.load(file)
    fig = plt.figure()
    img = plt.imshow(denorm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title('DENORMALISED')
    plt.colorbar()
    plt.show()
    plt.close()

    spectrogram = 10 ** (denorm_spectrogram / 10) - 1e-1
    reconstructed = librosa.griffinlim(spectrogram, n_iter=1000, hop_length=128)
    scipy.io.wavfile.write(generated_vocal_audio + 'REVERSED_' + name + '.wav', SR, reconstructed)

# ---------------------------------------------------------------------------------------------------------------------
# COMPARE GENERATED WAVEFORM WITH GROUND TRUTH
# ---------------------------------------------------------------------------------------------------------------------


generated_vocal, _ = librosa.load(generated_vocal_audio +
                                  "REVERSED_GENERATED_normalised_new_flute_acoustic_002-091-050_STFTMAG_NEW.wav",
                                  res_type='kaiser_fast', mono=True, sr=None)

fig, ax = plt.subplots(nrows=3)

librosa.display.waveshow(vocal[0:32640], sr=SR, ax=ax[0])
ax[0].set(title='ORIGINAL VOCAL')
ax[0].label_outer()

librosa.display.waveshow(generated_vocal, sr=SR, ax=ax[1])
ax[1].set(title='GENERATED VOCAL')
ax[1].label_outer()

librosa.display.waveshow(vocal[0:32640] - generated_vocal, sr=SR, ax=ax[2])
ax[2].set(title='DIFFERENCE')
ax[2].label_outer()

plt.tight_layout()
plt.show()
plt.close()

print("MSE(resyntesized audio, original audio): ", np.mean((vocal[0:32640] - generated_vocal[0:32640]) ** 2))
print("NMSE(resyntesized audio, original audio): ", np.mean((vocal[0:32640] - generated_vocal[0:32640]) ** 2 /
      vocal[0:32640] ** 2))

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE THE SPECTROGRAM OF THE RESYNTHESIZED AUDIO TO COMPARE WITH GROUND TRUTH
# ---------------------------------------------------------------------------------------------------------------------

feature_calculation(generated_vocal_audio, generated_vocal_features)

original_vocal = pathlib.Path(vocal_features).iterdir()
resynthesized_vocal = pathlib.Path(generated_vocal_features).iterdir()
for original, generated in zip(original_vocal, resynthesized_vocal):
    original_spectrogram = np.load(original)[0:512, 0:256]
    resynthesized_spectrogram = np.load(generated)[0:512, 0:256]
    print("MSE(resynthesized spectrogram, original spectrogram): ",
          np.mean((original_spectrogram - resynthesized_spectrogram) ** 2))
    print("NMSE(resynthesized spectrogram, original spectrogram): ",
          np.mean((original_spectrogram - resynthesized_spectrogram) ** 2 / original_spectrogram ** 2))
