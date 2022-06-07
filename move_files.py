import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/prova_copy/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/prova_paste/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)

