import os
import numpy as np
import pathlib
import shutil
import math
import librosa
import librosa.display
import scipy.io.wavfile
from functions import rename_files_by_pitch, count_pitches, append_pitches_velocities,\
    remove_files_if_pitch_not_matching, how_many_pitches

nsynth_train_path_subset_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/flute_acoustic'
nsynth_train_path_subset_vocal = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/vocal_synthetic'


flute_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_flute)
vocal_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_vocal)

not_present_A = []
not_present_B = []

for flute_pitch_velocity in sorted(flute_pitches_velocities):
    if flute_pitch_velocity not in vocal_pitches_velocities:
        not_present_A.append(flute_pitch_velocity)

for vocal_pitch_velocity in sorted(vocal_pitches_velocities):
    if vocal_pitch_velocity not in flute_pitches_velocities:
        not_present_B.append(vocal_pitch_velocity)


def pitches_count(path):
    pitches = []
    files_dir = pathlib.Path(path)
    files_in_basepath = files_dir.iterdir()
    for file in sorted(files_in_basepath):
        name = file.name
        pitch = name[0:3]
        pitches.append(pitch)
    return sorted(set(pitches))


flute_pitches = pitches_count(nsynth_train_path_subset_flute)
vocal_pitches = pitches_count(nsynth_train_path_subset_vocal)


flute_counts = how_many_pitches(nsynth_train_path_subset_flute)
vocal_counts = how_many_pitches(nsynth_train_path_subset_vocal)

print('sum(vocal_counts):', sum(vocal_counts))

print('debug')


# TODO
"""
eliminare quelli che non hanno tutte le velocities. NO
Rinominare i file per pitch e fare un count di quanti pitch ho dell'uno e dell'altro
Far matchare il numero di pitches di uno e dell'altro in modo da trainare ordinatamente
"""