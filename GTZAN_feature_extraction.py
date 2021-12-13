import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa

path_images = "/nas/home/spol/Thesis/GTZAN/images/"
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
for genre in genres:
    try:
        os.mkdir(path_images + genre)
    except OSError:
        print("Creation of the directory %s failed" % path_images + genre)
    else:
        print("Successfully created the directory %s " % path_images + genre)

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


for names in genres:
    data_dir = pathlib.Path('GTZAN/genres_original' + '/' + names)
    files_in_basepath = data_dir.iterdir()
    print(str(files_in_basepath))
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
            fig.savefig(path_images + names + name)
        else:
            print('thats nota a file')
