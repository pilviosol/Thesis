
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import librosa.display
import librosa


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
'''for item in files_in_basepath:
    if item.is_file():
        print(item.name)
    features = extract_features(item)'''

sample = pathlib.Path('GTZAN/genres_original/blues/blues.00000.wav')
features = extract_features(sample)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(features, ref=np.max),
                               sr=_SAMPLING_RATE, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()