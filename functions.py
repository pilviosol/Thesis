import os
import numpy as np
import pathlib
import shutil
import math


# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
CONSTANT_256 = 256


def divider_256(origin_path, destination_path):
    """

    :param origin_path: path where the features are extracted
    :param destination_path: path where to store the features divided in 256 long chunks
    :return: void

    """
    features_dir = pathlib.Path(origin_path)
    features_in_basepath = features_dir.iterdir()

    for item in features_in_basepath:
        name = item.name
        print(name)
        feature = np.load(item)
        length = feature.shape[1]
        chunks = math.floor(length / CONSTANT_256)
        print(chunks)
        for i in range(chunks):
            chunk = feature[0:1025, i * CONSTANT_256: (i + 1) * CONSTANT_256]
            print("chunk.shape: ", chunk.shape)
            np.save(destination_path + name + "_chunk_" + str(i), chunk)
            print("i: ", i)
        print('--------------------------------------------')
