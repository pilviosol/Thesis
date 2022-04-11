from VV_autoencoder import VAE
from functions import feature_calculation, denormalise, fw_normalise, min_max_array_saving
import matplotlib.pyplot as plt
import librosa.display
import pathlib
import numpy as np
import scipy.io.wavfile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from WANDB import config
import os

# ---------------------------------------------------------------------------------------------------------------------
# PATH, VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
with open('/nas/home/spol/Thesis/last_date.txt') as f:
    date = f.read()
    print('date: ', date)

main_folder = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/" + date + "/"
os.mkdir(main_folder)

matching_flute_TEST = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_flute_TEST/"

features_matching_flute_TEST = main_folder + "features_matching_flute_TEST/"
os.mkdir(features_matching_flute_TEST)

matching_vocal_TEST = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_vocal_TEST/"

features_matching_vocal_TEST = main_folder + "features_matching_vocal_TEST/"
os.mkdir(features_matching_vocal_TEST)

Figures_TEST = main_folder + "Figures_TEST/"
os.mkdir(Figures_TEST)

normalised_flute_features_TEST = main_folder + "normalised_flute_features_TEST/"
os.mkdir(normalised_flute_features_TEST)

normalised_vocal_features_TEST = main_folder + "normalised_vocal_features_TEST/"
os.mkdir(normalised_vocal_features_TEST)

generated_vocal_features_TEST = main_folder + "generated_vocal_features_TEST/"
os.mkdir(generated_vocal_features_TEST)

# min_max_flute_inference_path_file = main_folder + "flute_min_max_TEST.npy"
min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/flute_"
min_max_vocal_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/vocal_"


denormalised_spectrogram_TEST = main_folder + "denormalised_spectrogram_TEST/"
os.mkdir(denormalised_spectrogram_TEST)

generated_vocal_TEST = main_folder + "generated_vocal_TEST/"
os.mkdir(generated_vocal_TEST)

generated_vocal_features = main_folder + "vocal_generated_features/"
os.mkdir(generated_vocal_features)

os.mkdir(Figures_TEST + 'input-output-expected_output/')
os.mkdir(Figures_TEST + 'denormalised-output/')
os.mkdir(Figures_TEST + 'original-generated-difference/')

SR = config['sample_rate']
print('PATH, VARIABLES..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------


feature_calculation(matching_flute_TEST, features_matching_flute_TEST)
feature_calculation(matching_vocal_TEST, features_matching_vocal_TEST)

print('CALCULATE SPECTROGRAMS..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
min_max_flute = min_max_array_saving(features_matching_flute_TEST, min_max_flute_path_file)
flute_folder_min_max = fw_normalise(features_matching_flute_TEST, normalised_flute_features_TEST, min_max_flute)
# min_max_flute = normalise_set_and_save_min_max(features_matching_flute_TEST, normalised_flute_features_TEST)

min_max_vocal = min_max_array_saving(features_matching_vocal_TEST, min_max_vocal_path_file)
vocal_folder_min_max = fw_normalise(features_matching_vocal_TEST, normalised_vocal_features_TEST, min_max_vocal)

print('NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL AND DEFINE GENERATE FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)

def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations

print('IMPORT THE MODEL..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# GENERATE NEW SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------


normalised_features_path = pathlib.Path(normalised_flute_features_TEST).iterdir()
for file in sorted(normalised_features_path):
    name = file.name
    spectrogram = np.load(file)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print('spectrogram expanded dims: ', spectrogram.shape)

    generated_spectrogram, latent_representation = generate(spectrogram)
    np.save(generated_vocal_features_TEST + 'GENERATED_' + name, generated_spectrogram)

print('FEED THEM TO THE MODEL AND GENERATE NEW SPECTROGRAMS..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# PLOT INPUT, OUTPUT and EXPECTED OUTPUT OF THE NETWORK
# ---------------------------------------------------------------------------------------------------------------------
VAE_input = pathlib.Path(normalised_flute_features_TEST).iterdir()
VAE_output = pathlib.Path(generated_vocal_features_TEST).iterdir()
VAE_expected_output = pathlib.Path(normalised_vocal_features_TEST).iterdir()

for inpt, output, expected_output in zip(sorted(VAE_input), sorted(VAE_output), sorted(VAE_expected_output)):
    input_name = inpt.name
    print('input_name: ', input_name)
    input_spectrogram = np.load(inpt)
    print('input_spectrogram.shape: ', input_spectrogram.shape)

    output_name = output.name
    print('output_name: ', output_name)
    output_spectrogram = np.load(output)
    out_spectrogram = np.squeeze(output_spectrogram)
    print('out_spectrogram.shape: ', out_spectrogram.shape)

    expected_output_name = expected_output.name
    expected_output_spectrogram = np.load(expected_output)
    print('expected_output_spectrogram dims: ', expected_output_spectrogram.shape)

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # VAE INPUT
    img1 = ax1.imshow(input_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax1.set_title('INPUT_' + input_name[34:-16])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    # VAE OUTPUT
    img2 = ax2.imshow(out_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax2.set_title('OUTPUT_' + output_name[44:-16])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)

    # EXPECTED OUTPUT
    img3 = ax3.imshow(expected_output_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                      aspect='auto')
    ax3.set_title('EXPECTED OUTPUT' + expected_output_name[44:-16])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(img3, cax=cax3)

    plt.tight_layout()
    plt.savefig(Figures_TEST + 'input-output-expected_output/' + input_name[33:-16])

    plt.show()
    plt.close()

    print("MSE(net out spec, expected out spec): ", np.mean((out_spectrogram - expected_output_spectrogram) ** 2))
    print("NMSE(net out spec, expected out spec): ", np.mean((out_spectrogram - expected_output_spectrogram) ** 2 /
                                                             expected_output_spectrogram ** 2))

print('PLOT THE OUTPUT OF THE NETWORK..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE THEM
# ---------------------------------------------------------------------------------------------------------------------

# min_max_values = np.load(min_max_flute_inference_path_file)
# min_max_values = flute_folder_min_max

generated_spectrograms_path = pathlib.Path(generated_vocal_features_TEST).iterdir()
for file in sorted(generated_spectrograms_path):
    name = file.name
    gen_spectrogram = np.load(file)
    gen_spectrogram = np.squeeze(gen_spectrogram)
    # denormalised_spectrogram = denormalise_given_min_max(gen_spectrogram, min_max_values[0][0], min_max_values[0][1])
    denormalised_spectrogram = denormalise(gen_spectrogram, flute_folder_min_max[0][1], flute_folder_min_max[0][0])
    np.save(denormalised_spectrogram_TEST + name, denormalised_spectrogram)

print('GET THE OUTPUT AND DE-NORMALISE THEM..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAMS AND RESYNTHESIZE THEM
# ---------------------------------------------------------------------------------------------------------------------
denormalised_generated_spectrogram_path = pathlib.Path(denormalised_spectrogram_TEST).iterdir()
for file in sorted(denormalised_generated_spectrogram_path):
    name = file.name
    print(name)

    denorm_spectrogram = np.load(file)

    fig = plt.figure()
    img = plt.imshow(denorm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title('DENORMALIZED_' + name[43:-16])
    plt.colorbar()
    plt.savefig(Figures_TEST + 'denormalised-output/' + name[44:-20])
    plt.show()
    plt.close()

    spectrogram = 10 ** (denorm_spectrogram / 10) - 1e-5
    reconstructed = librosa.griffinlim(spectrogram, n_iter=1000, hop_length=128)
    scipy.io.wavfile.write(generated_vocal_TEST + 'REVERSED_' + name + '.wav', SR, reconstructed)

print('PLOT THE RECONSTRUCTED SPECTROGRAMS..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# COMPARE GENERATED WAVEFORM WITH GROUND TRUTH
# ---------------------------------------------------------------------------------------------------------------------
vocals = pathlib.Path(matching_vocal_TEST).iterdir()
generated_vocal_path = pathlib.Path(generated_vocal_TEST).iterdir()
for vocal, generated_vocal in zip(sorted(vocals), sorted(generated_vocal_path)):
    name = vocal.name
    name = name[0:-4]
    print('generated_vocal_TEST.name: ', name)
    vocal, _ = librosa.load(vocal,  res_type='kaiser_fast', mono=True, sr=None)
    generated_vocal, _ = librosa.load(generated_vocal,  res_type='kaiser_fast', mono=True, sr=None)

    fig, ax = plt.subplots(nrows=3)

    librosa.display.waveshow(vocal[0:32640], sr=SR, ax=ax[0])
    ax[0].set(title='ORIGINAL VOCAL')
    ax[0].label_outer()

    librosa.display.waveshow(generated_vocal[0:32640], sr=SR, ax=ax[1])
    ax[1].set(title='GENERATED VOCAL')
    ax[1].label_outer()

    librosa.display.waveshow(vocal[0:32640] - generated_vocal[0:32640], sr=SR, ax=ax[2])
    ax[2].set(title='DIFFERENCE')
    ax[2].label_outer()

    plt.tight_layout()
    plt.savefig(Figures_TEST + 'original-generated-difference/' + name)
    plt.show()
    plt.close()

    print("MSE(resyntesized audio, original audio): ", np.mean((vocal[0:32640] - generated_vocal[0:32640]) ** 2))
    print("NMSE(resyntesized audio, original audio): ", np.mean((vocal[0:32640] - generated_vocal[0:32640]) ** 2 /
          vocal[0:32640] ** 2))

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE THE SPECTROGRAM OF THE RESYNTHESIZED AUDIO TO COMPARE WITH GROUND TRUTH
# ---------------------------------------------------------------------------------------------------------------------

feature_calculation(generated_vocal_TEST, generated_vocal_features)

original_vocal = pathlib.Path(features_matching_vocal_TEST).iterdir()
resynthesized_vocal = pathlib.Path(generated_vocal_features).iterdir()
for original, generated in zip(original_vocal, resynthesized_vocal):
    original_spectrogram = np.load(original)[0:512, 0:256]
    resynthesized_spectrogram = np.load(generated)[0:512, 0:256]
    print("MSE(resynthesized spectrogram, original spectrogram): ",
          np.mean((original_spectrogram - resynthesized_spectrogram) ** 2))
    print("NMSE(resynthesized spectrogram, original spectrogram): ",
          np.mean((original_spectrogram - resynthesized_spectrogram) ** 2 / original_spectrogram ** 2))


print('debug')
