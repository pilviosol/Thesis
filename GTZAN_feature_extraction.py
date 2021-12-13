import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa
import shutil

path_images = "/nas/home/spol/Thesis/GTZAN/images/"
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

try:
    shutil.rmtree(path_images, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % path_images)
else:
    print("Successfully removed the directory %s" % path_images)

for genre in genres:
    print(path_images + genre)
    try:
        os.makedirs(path_images + genre + "/train")
        os.makedirs(path_images + genre + "/test")
    except OSError:
        print("Creation of the directory %s failed" % (path_images + genre))
    else:
        print("Successfully created the directory %s" % (path_images + genre))

_SAMPLING_RATE = 22000
print('ollare')
data_dir = pathlib.Path('GTZAN/genres_original')


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', mono=True)
        cqt = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return cqt


for genre in genres:
    data_dir = pathlib.Path('GTZAN/genres_original' + '/' + genre)
    files_in_basepath = data_dir.iterdir()
    count = 0
    for item in files_in_basepath:
        if item.is_file():
            print(item.name)
            features = extract_features(item)
            name = item.name[0:-4] + '_CQT.png'
            fig, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(features, ref=np.max),
                                           sr=_SAMPLING_RATE, x_axis='time', y_axis='cqt_note', ax=ax)
            ax.set_title(name)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            if count < 80:
                fig.savefig(path_images + genre + "/train/" + name)
            else:
                fig.savefig(path_images + genre + "/test/" + name)
        else:
            print('That is not a file')
        count += 1
