import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa

path = "/nas/home/spol/Thesis/GTZAN/images"

try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

_SAMPLING_RATE = 16000
print('ollare')
data_dir = pathlib.Path('GTZAN/genres_original/blues')


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', mono=True)
        cqt = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return cqt


files_in_basepath = data_dir.iterdir()
i = 0
for item in files_in_basepath:

    if item.is_file():
        print(item.name)
        features = extract_features(item)
        name = 'spectrogram' + str(i)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(features, ref=np.max),
                                       sr=_SAMPLING_RATE, x_axis='time', y_axis='cqt_note', ax=ax)
        ax.set_title('Constant-Q power spectrum')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.savefig('/nas/home/spol/Thesis/GTZAN/images/' + name)
    i = i + 1









# sample = pathlib.Path('GTZAN/genres_original/blues/blues.00000.wav')

# plt.show()
