import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf
from tensorflow.keras.models import load_model

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
#  PREVISIONI
# ---------------------------------------------------------------
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

generated_spectrograms = load_fsdd(path)

predictions = probability_model.predict(generated_spectrograms)


for i in range(352):
    print(i, 'predictions: ', np.around(predictions[i], 3), 'class: ', np.argmax(predictions[i]))