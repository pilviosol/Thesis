import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/NEW_HQ_matching_flute_TRAIN/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/SUPER_HQ_MATCHING_FLUTE_TRAIN/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)