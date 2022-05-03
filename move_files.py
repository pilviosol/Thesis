import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/keyboard_acoustic/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/matching_keyboard_VALID/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)