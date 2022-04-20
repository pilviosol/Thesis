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

nsynth_train_path_subset_flute = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/HQ_matching_flute_TRAIN'
nsynth_train_path_subset_string = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/HQ_matching_string_TRAIN'

matching_flute_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_matching_flute_TRAIN'
matching_string_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_matching_string_TRAIN'


flute_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_flute)
vocal_pitches_velocities = append_pitches_velocities(nsynth_train_path_subset_string)

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
string_pitches = pitches_count(nsynth_train_path_subset_string)


flute_counts = how_many_pitches(nsynth_train_path_subset_flute)
string_counts = how_many_pitches(nsynth_train_path_subset_string)

print('sum(vocal_counts):', sum(string_counts))
print('sum(flute_counts):', sum(flute_counts))




matching = []
for i, flute_count in enumerate(flute_counts):
    if flute_counts[i] < string_counts[i]:
        matching.append(flute_counts[i])
    else:
        matching.append(string_counts[i])

print('sum(matching):', sum(matching))


'''
Here you have to put the 2 folders to be defined i.e. matching_flute and matching_string
Watch out for line "if name[0:3] == str(string_pitches[idx])" and
"shutil.copy(song, matching_string_path)" that also have to be changed
'''
for idx, count in enumerate(matching):
    files_dir = pathlib.Path(nsynth_train_path_subset_string)
    files_in_basepath = files_dir.iterdir()
    print('---------------count: ', count)
    print('--idx: ', idx)
    c = 0
    for song in sorted(files_in_basepath):
        name = song.name
        print('name: ', name)
        if name[0:3] == str(string_pitches[idx]):
            print('name[0:3]', name[0:3])
            print('str(string_pitches[idx]):', str(string_pitches[idx]))
            print('copying file')
            shutil.copy(song, matching_string_path)
            c += 1
            print('c: ', c)
            if c == count:
                print('break')
                break


print('debug')


