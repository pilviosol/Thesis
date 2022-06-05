import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------
model_path = "/nas/home/spol/Thesis/Classifier.h5"

path_01 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/01/"
path_12 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/12/"
path_23 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/23/"
path_30 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/30/"

path_col1 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/col1/"
path_col2 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/col2/"
path_row1 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/row1/"
path_row2 = "/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/INTERPOLATION_multi/18052022_2237_savings/INTERPOLATIONs/row2/"
# ---------------------------------------------------------------
#  LOAD CLASSIFIER
# ---------------------------------------------------------------

model = load_model(model_path)

# ---------------------------------------------------------------
#  PREVISIONI
# ---------------------------------------------------------------
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

interpolation_12 = load_fsdd(path_30)

predictions = probability_model.predict(interpolation_12)


for i in range(10):
    print(i, 'predictions: ', predictions[i], 'class: ', np.argmax(predictions[i]))