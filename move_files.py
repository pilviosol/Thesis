import shutil
import pathlib

origin_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/WAV_flute/"
destination_path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/temp/"

origin = pathlib.Path(origin_path).iterdir()
destination = pathlib.Path(destination_path)


for file in origin:
    shutil.copy(file, destination_path)

