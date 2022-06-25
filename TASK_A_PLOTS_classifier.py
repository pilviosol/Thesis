import matplotlib.pyplot as plt
import numpy as np
import pathlib
from functions import load_fsdd, PLOT_PRINT_OPTIONS
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.colors as colors


# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------
model_path = "/nas/home/spol/Thesis/Classifier_256.h5"
conv_model_path = "/nas/home/spol/Thesis/Conv_classifier_256.h5"
save_path = "/nas/home/spol/Thesis/plots_thesis/"


path =  "/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/GENERATED_SPECTROGRAMS/GENERATED_SPECTROGRAMS_0906_0101/"
# ---------------------------------------------------------------
#  LOAD CLASSIFIER
# ---------------------------------------------------------------

model = load_model(conv_model_path)

# ---------------------------------------------------------------
#  LABELS
# ---------------------------------------------------------------
classes = [0, 1, 2, 3]

y_val = []
for i in range(352):
    if i < 88:
        y_val.append(classes[2])
    elif 88 <= i < 176:
        y_val.append(classes[1])
    elif 176 <= i < 264:
        y_val.append(classes[3])
    else:
        y_val.append(classes[0])
y_val = np.asarray(y_val)

print(y_val)

labels=["string", "keyboard", "guitar", "organ"]
# ---------------------------------------------------------------
#  PREVISIONI
# ---------------------------------------------------------------
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

generated_spectrograms = load_fsdd(path)

predictions = probability_model.predict(generated_spectrograms)
arg_pred_all = []

for i in range(352):
    # print(i, 'predictions: ', np.around(predictions[i], 3), 'class: ', np.argmax(predictions[i]))
    arg_pred_all.append(np.argmax(predictions[i]))

print(arg_pred_all)

y_true = y_val
y_pred = np.asarray(arg_pred_all)
cm = confusion_matrix(y_true, y_pred, labels=classes)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('Set1')
my_cmap = truncate_colormap(cmap, 0, 0.33, 4)


fig, ax = plt.subplots()
fig = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=("S", "K", "G", "O") )
fig.plot(cmap='Blues', colorbar=False)
plt.savefig(save_path + 'confusion_matrix_Blues', **PLOT_PRINT_OPTIONS)
plt.show()