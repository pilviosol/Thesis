import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------
model_path = "/nas/home/spol/Thesis/Classifier.h5"
conv_model_path = "/nas/home/spol/Thesis/Conv_classifier.h5"


path = "/nas/home/spol/Thesis/INTERPOLATIONS/07062022_2/INTERPOLATIONs/column3/"
# ---------------------------------------------------------------
#  LOAD CLASSIFIER
# ---------------------------------------------------------------

model = load_model(conv_model_path)

# ---------------------------------------------------------------
#  PREVISIONI
# ---------------------------------------------------------------
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

interpolations = load_fsdd(path)

predictions = probability_model.predict(interpolations)


for i in range(5):
    print(i, 'predictions: ', np.around(predictions[i], 3), 'class: ', np.argmax(predictions[i]))