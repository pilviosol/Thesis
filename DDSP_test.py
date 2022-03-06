from ddsp_main.ddsp.training.metrics import compute_audio_features
from ddsp_main.ddsp.spectral_ops import reset_crepe
from ddsp_main.ddsp.training.models import autoencoder
import warnings
import copy
import os
import time
import crepe
import ddsp
import librosa
import matplotlib.pyplot as plt
import numpy as npimport
import pickle

# import tensorflow_datasets as tfds
warnings.filterwarnings("ignore")


# Helper Functions
DEFAULT_SAMPLE_RATE = 16000
sample_rate = DEFAULT_SAMPLE_RATE  # 16000

print('Done!')


# --------------------------------------------------------------------
audio, sr = librosa.load('scala_sustained.wav', sr=DEFAULT_SAMPLE_RATE)
if len(audio.shape) == 1:
    audio = audio[np.newaxis, :]
print('\nExtracting audio features...')


# Setup the session.
reset_crepe()


# Compute features.
start_time = time.time()

audio_features = compute_audio_features(audio)
# audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
audio_features_mod = None
print('Audio features took %.1f seconds' % (time.time() - start_time))


TRIM = -15
# Plot Features.
fig, ax = plt.subplots(nrows=3, 
                       ncols=1, 
                       sharex=True,
                       figsize=(6, 8))
ax[0].plot(audio_features['loudness_db'][:TRIM])
ax[0].set_ylabel('loudness_db')

ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
ax[1].set_ylabel('f0 [midi]')

ax[2].plot(audio_features['f0_confidence'][:TRIM])
ax[2].set_ylabel('f0 confidence')
_ = ax[2].set_xlabel('Time step [frame]')

plt.show()


# --------------------------------------------------------------------

# VV_model = 'Violin'
# MODEL = VV_model

# hop_size = int(n_samples_train / time_steps_train)
hop_size = 256

time_steps = int(audio.shape[1] / hop_size)
n_samples = time_steps * hop_size


# Set up the VV_model just to predict audio given new conditioning
model = autoencoder.Autoencoder()

# Build VV_model by running a batch through it.
start_time = time.time()
print('Restoring VV_model took %.1f seconds' % (time.time() - start_time))


# --------------------------------------------------------------------

threshold = 1
ADJUST = True
quiet = 20
autotune = 0
pitch_shift = 0
loudness_shift = 0


# --------------------------------------------------------------------
af = audio_features

# Run a batch of predictions.
start_time = time.time()
outputs = model(af, training=False)
audio_gen = model.get_audio_from_outputs(outputs)
print('Prediction took %.1f seconds' % (time.time() - start_time))
