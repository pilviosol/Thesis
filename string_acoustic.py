from functions import rename_files_by_pitch
import pathlib
import os

path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/string_acoustic"
a_path = pathlib.Path(path).iterdir()

rename_files_by_pitch(path)
for element in sorted(a_path):
    name = element.name
    pitch_name = name[0:3]
    if int(pitch_name) < 21 or int(pitch_name) > 107:
        os.remove(path + '/' + name)
