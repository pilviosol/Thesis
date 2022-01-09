import os
import numpy as np
import pathlib

# DIVIDE THE VIOLIN FEATURES IN 4 SECONDS CHUNKS

print('Dividing vn features in 4 seconds chunks....')
features_dir_vn = pathlib.Path('URPM_vn_fl/features_vn')
features_in_basepath_vn = features_dir_vn.iterdir()


for item in features_in_basepath_vn:
    print(item.name)
    feature = np.load(item)
    print(feature.shape)
    print('--------------------------------------------')























'''
# DIVIDE THE FLUTE FEATURES IN 4 SECONDS CHUNKS

print('Dividing vn features in 4 seconds chunks....')
features_dir_fl = pathlib.Path('URPM_vn_fl/features_fl')
features_in_basepath_fl = features_dir_fl.iterdir()
print('ollare fl')

for item in features_in_basepath_fl:
    print(item.name) '''
