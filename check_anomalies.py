import pathlib
import numpy as np

folder_to_be_checked = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/FW_normalised_vocal"
path = pathlib.Path(folder_to_be_checked).iterdir()

maximum = []
minimum = []
for file in sorted(path):
    name = file.name
    print(name)
    loaded = np.load(file)
    maximum.append(loaded.max())
    minimum.append(loaded.min())
maximum = np.array(maximum)
minimum = np.array(minimum)
print('debug')

