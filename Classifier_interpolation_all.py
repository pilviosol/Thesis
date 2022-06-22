import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------
main_path = "/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/"
model_path = "/nas/home/spol/Thesis/Classifier_256.h5"
conv_model_path = "/nas/home/spol/Thesis/Conv_classifier_256.h5"
conv_reducted_model_path = "/nas/home/spol/Thesis/Conv_classifier_reducted.h5"


KS_path = main_path + "KS/"
KG_path = main_path + "KG/"
GO_path = main_path + "GO/"
SO_path = main_path + "SO/"


# ---------------------------------------------------------------
#  LOAD CLASSIFIER
# ---------------------------------------------------------------

model = load_model(conv_reducted_model_path)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# ---------------------------------------------------------------
#  KS PREDICTIONS
# ---------------------------------------------------------------

interpolations = load_fsdd(KS_path)

predictions = probability_model.predict(interpolations)

print('debug')




