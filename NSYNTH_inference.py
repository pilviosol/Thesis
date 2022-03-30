from VV_autoencoder import VAE
from functions import feature_calculation, denormalise, normalise_set_and_save_min_max
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

generated_vocal_features_TEST = main_folder + "generated_vocal_features_TEST/"
os.mkdir(generated_vocal_features_TEST)

min_max_flute_inference_path_file = main_folder + "flute_min_max_TEST.npy"
# min_max_flute_path_file = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/flute_"

denormalised_spectrogram_TEST = main_folder + "denormalised_spectrogram_TEST/"
os.mkdir(denormalised_spectrogram_TEST)

generated_vocal_TEST = main_folder + "generated_vocal_TEST/"
os.mkdir(generated_vocal_TEST)

os.mkdir(Figures_TEST + 'input-output/')
os.mkdir(Figures_TEST + 'input-expected_output/')
os.mkdir(Figures_TEST + 'denormalised-output/')

SR = config['sample_rate']
print('PATH, VARIABLES..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# CALCULATE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
feature_calculation(matching_flute_TEST, features_matching_flute_TEST)
feature_calculation(matching_vocal_TEST, features_matching_vocal_TEST)

print('CALCULATE SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------

flutes_path = pathlib.Path(features_matching_flute_TEST).iterdir()
vocals_path = pathlib.Path(features_matching_vocal_TEST).iterdir()

for flute, vocal in zip(sorted(flutes_path), sorted(vocals_path)):
    flute_name = flute.name
    print('flute_name[19:-16]: ', flute_name[19:-16])
    flute_spectrogram = np.load(flute)
    flute_spectrogram = flute_spectrogram[0:512, 0:256]
    vocal_name = vocal.name
    vocal_spectrogram = np.load(vocal)
    vocal_spectrogram = vocal_spectrogram[0:512, 0:256]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # FLUTE SPECTROGRAMS
    img1 = ax1.imshow(flute_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax1.set_title('flute_' + flute_name[19:-16])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(img1, cax=cax1)

    # CORRESPONDING FLUTE SPECTROGRAMS (WHAT WE WOUKD WANT AS OUTPUT)
    img2 = ax2.imshow(vocal_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax2.set_title('vocal_' + vocal_name[20:-16])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(img2, cax=cax2)
    plt.savefig(Figures_TEST + 'input-expected_output/' + flute_name[19:-16])
    # plt.show()

    plt.close()

print('PLOT THE SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX
# ---------------------------------------------------------------------------------------------------------------------
# min_max_flute = min_max_array_saving(features_matching_flute_TEST, min_max_flute_path_file)
# flute_folder_min_max = fw_normalise(features_matching_flute_TEST, normalised_flute_features_TEST, min_max_flute)
min_max_flute = normalise_set_and_save_min_max(features_matching_flute_TEST, normalised_flute_features_TEST)

print('NORMALIZE THE SPECTROGRAMS AND SAVE MIN MAX..........ok')

'''
# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE NORMALIZED SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
files_dir = pathlib.Path(normalised_flute_features_TEST)
files_in_basepath = files_dir.iterdir()
for file in files_in_basepath:
    name = file.name
    spectrogram = np.load(file)
    spectrogram = spectrogram[0:512, 0:256]

    fig = plt.figure()
    img = plt.imshow(spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512],
                     aspect='auto')
    plt.title('NORMALIZED' + name)
    plt.colorbar()
    # plt.show()
    plt.close()

print('PLOT THE NORMALIZED SPECTROGRAMS..........ok')
'''
# ---------------------------------------------------------------------------------------------------------------------
# IMPORT THE MODEL
# ---------------------------------------------------------------------------------------------------------------------
# il migliore ad oggi:
# vae = VAE.load("/nas/home/spol/Thesis/saved_model/VV_model_FW_x_train_y_train_x_val_y_val")
vae = VAE.load("/nas/home/spol/Thesis/saved_model/" + date)


print('IMPORT THE MODEL..........ok')


# ---------------------------------------------------------------------------------------------------------------------
# FEED THEM TO THE MODEL AND GENERATE NEW SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
def generate(spectrograms):
    generated_spectrograms, latent_representations = vae.reconstruct(spectrograms)
    return generated_spectrograms, latent_representations


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
# PLOT THE INPUT AND OUTPUT OF THE NETWORK
# ---------------------------------------------------------------------------------------------------------------------
VAE_input = pathlib.Path(normalised_flute_features_TEST).iterdir()
VAE_output = pathlib.Path(generated_vocal_features_TEST).iterdir()

for inpt, output in zip(sorted(VAE_input), sorted(VAE_output)):
    input_name = inpt.name
    print('input_name: ', input_name)
    input_spectrogram = np.load(inpt)
    input_spectrogram = input_spectrogram[0:512, 0:256]
    output_name = output.name
    print('output_name: ', output_name)
    output_spectrogram = np.load(output)
    output_spectrogram = output_spectrogram[0:512, 0:256]
    out_spectrogram = np.squeeze(output_spectrogram)

    fig2, (ax3, ax4) = plt.subplots(1, 2)

    # VAE INPUT
    img3 = ax3.imshow(input_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax3.set_title('INPUT_' + input_name[34:-16])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(img3, cax=cax3)

    # VAE OUTPUT
    img4 = ax4.imshow(out_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    ax4.set_title('OUTPUT_' + output_name[44:-16])
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(img4, cax=cax4)
    plt.savefig(Figures_TEST + 'input-output/' + input_name[34:-16])

    plt.show()
    plt.close()

print('PLOT THE OUTPUT OF THE NETWORK..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# GET THE OUTPUT AND DE-NORMALISE THEM
# ---------------------------------------------------------------------------------------------------------------------

# min_max_values = np.load(min_max_flute_inference_path_file)
# min_max_values = flute_folder_min_max

generated_spectrograms_path = pathlib.Path(generated_vocal_features_TEST).iterdir()
for idx, file in enumerate(sorted(generated_spectrograms_path)):
    name = file.name
    gen_spectrogram = np.load(file)
    gen_spectrogram = np.squeeze(gen_spectrogram)
    # denormalised_spectrogram = denormalise_given_min_max(gen_spectrogram, min_max_values[0][0], min_max_values[0][1])
    denormalised_spectrogram = denormalise(gen_spectrogram, min_max_flute[idx][0], min_max_flute[idx][1])
    np.save(denormalised_spectrogram_TEST + name, denormalised_spectrogram)

print('GET THE OUTPUT AND DE-NORMALISE THEM..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# PLOT THE RECONSTRUCTED SPECTROGRAMS
# ---------------------------------------------------------------------------------------------------------------------
denormalised_generated_spectrogram_path = pathlib.Path(denormalised_spectrogram_TEST).iterdir()
for file in sorted(denormalised_generated_spectrogram_path):
    name = file.name
    print(name)
    denorm_spectrogram = np.load(file)
    # denorm_spectrogram = 10 ** (denorm_spectrogram / 10) - 1e-1
    fig = plt.figure()
    img = plt.imshow(denorm_spectrogram, cmap=plt.cm.viridis, origin='lower', extent=[0, 256, 0, 512], aspect='auto')
    plt.title('DENORMALIZED_' + name[44:-16])
    plt.colorbar()
    plt.savefig(Figures_TEST + 'denormalised-output/' + name[44:-20])
    plt.show()
    plt.close()

print('PLOT THE RECONSTRUCTED SPECTROGRAMS..........ok')

# ---------------------------------------------------------------------------------------------------------------------
# SYNTHESIZE THE NEW AUDIO
# ---------------------------------------------------------------------------------------------------------------------
denormalised_generated_spectrogram_path = pathlib.Path(denormalised_spectrogram_TEST).iterdir()
for file in sorted(denormalised_generated_spectrogram_path):
    name = file.name
    name = name[0:-4]
    print(name)
    denorm_spectrogram = np.load(file)
    reconstructed = librosa.griffinlim(denorm_spectrogram)
    scipy.io.wavfile.write(generated_vocal_TEST + 'REVERSED_' + name + '.wav', SR, reconstructed)

print('debug')
