import sys
sys.path.insert(1, '/ddsp_main/ddsp')
import ddsp_main.ddsp.core # così posso importare le funzioni che voglio cazzo
import warnings
import copy
import os
import time
import crepe
# import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
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

'''
# Compute features.
start_time = time.time()
audio_features = ddsp.training.metrics.compute_audio_features(audio)
audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
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


'''
