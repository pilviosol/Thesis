import pathlib
import numpy as np

folder_to_be_checked = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_flute"
path = pathlib.Path(folder_to_be_checked).iterdir()

maximum = []
maximum = np.array(maximum)
minimum = []
minimum = np.array(minimum)
for file in sorted(path):
    name = file.name
    print(name)
    loaded = np.load(file)
    maximum.append(loaded.max())
    minimum.append(loaded.min())

print('debug')