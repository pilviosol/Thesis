import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_flutes_TRAIN/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_flutes/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)