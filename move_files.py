import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/HQ_matching_flute_TRAIN/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/backup_flute/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)