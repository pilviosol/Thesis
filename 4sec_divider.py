import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa
import librosa.display

# DIVIDE THE VIOLIN FEATURES IN 4 SECONDS CHUNKS

print('Dividing vn features in 4 seconds chunks....')
features_dir_vn = pathlib.Path('URPM_vn_fl/features_vn')
features_in_basepath_vn = features_dir_vn.iterdir()

first_feature = next(features_in_basepath_vn)
print(first_feature)
feature = np.load(first_feature)
print(feature.shape[1])

second_feature = next(features_in_basepath_vn)
print(second_feature)
feature = np.load(second_feature)
print(feature.shape[1])

third_feature = next(features_in_basepath_vn)
print(third_feature)
feature = np.load(third_feature)
print(feature.shape[1])

'''
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(feature, ref=np.max),
                               sr=44100, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

for item in features_in_basepath_vn:
    print(item.name)
    feature = np.load(item)
    print(feature.shape)
    print('--------------------------------------------') '''























'''
# DIVIDE THE FLUTE FEATURES IN 4 SECONDS CHUNKS

print('Dividing vn features in 4 seconds chunks....')
features_dir_fl = pathlib.Path('URPM_vn_fl/features_fl')
features_in_basepath_fl = features_dir_fl.iterdir()
print('ollare fl')

for item in features_in_basepath_fl:
    print(item.name) '''
