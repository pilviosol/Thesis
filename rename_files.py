import os
import pathlib

path = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/"
path_iter = pathlib.Path(path).iterdir()

for element in path_iter:
    name = element.name
    length = len(name)
    print(name)

    a = name[0:14]
    print('a: ', a)
    pitch = name[-23:-20]
    print('pitch: ', pitch)
    c = name[13:-1]
    print('c: ', c)
    os.rename(path + name, path + a + pitch + c)

