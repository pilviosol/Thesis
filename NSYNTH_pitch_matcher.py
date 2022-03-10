import os
import numpy as np
import pathlib
import shutil
import math
import librosa
import librosa.display
import scipy.io.wavfile

nsynth_train_path_subset_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/flute_acoustic'
nsynth_train_path_subset_vocal = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/vocal_synthetic'


def append_pitches_velocities(path):
    pitches_velocities = []
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    for item in sorted(files_in_basepath):
        name = item.name
        pitch_velocity = str(name)[-11: -4]
        print('pitch_velocity', pitch_velocity)
        pitches_velocities.append(pitch_velocity)
    return pitches_velocities


flute_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_flute)
vocal_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_vocal)

not_present_A = []
not_present_B = []

for flute_pitch_velocity in flute_pitches_velocities:
    if flute_pitch_velocity not in vocal_pitches_velocities:
        not_present_A.append(flute_pitch_velocity)

for vocal_pitch_velocity in vocal_pitches_velocities:
    if vocal_pitch_velocity not in flute_pitches_velocities:
        not_present_B.append(vocal_pitch_velocity)

count = 0
for a in flute_pitches_velocities:
    if a[4:7] == '025':
        count += 1

print('debug')
