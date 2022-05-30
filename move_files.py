import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/OLD_SPLIT_reducted_flutes/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/OLD_SPLIT_flutes_backup/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)

