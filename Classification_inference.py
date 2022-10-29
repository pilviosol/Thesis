import matplotlib.pyplot as plt
import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------
model_path = "/nas/home/spol/Thesis/Classifier_256.h5"
conv_model_path = "/nas/home/spol/Thesis/Conv_classifier_256.h5"


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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()
