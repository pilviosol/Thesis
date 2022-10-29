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

KS_save_path = main_path + "KS_predictions"
KG_save_path = main_path + "KG_predictions"
GO_save_path = main_path + "GO_predictions"
SO_save_path = main_path + "SO_predictions"

# ---------------------------------------------------------------
#  LOAD CLASSIFIER
# ---------------------------------------------------------------

model = load_model(conv_reducted_model_path)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# ---------------------------------------------------------------
#  KS PREDICTIONS (ROW0)
# ---------------------------------------------------------------

ks_interpolations = load_fsdd(KS_path)
ks_interpolations = np.squeeze(ks_interpolations)
ks_interpolations = np.expand_dims(ks_interpolations, axis=-1)

ks_predictions = probability_model.predict(ks_interpolations)
np.save(KS_save_path, ks_predictions)

# ---------------------------------------------------------------
#  KG PREDICTIONS (COLUMN4)
# ---------------------------------------------------------------

kg_interpolations = load_fsdd(KG_path)
kg_interpolations = np.squeeze(kg_interpolations)
kg_interpolations = np.expand_dims(kg_interpolations, axis=-1)

kg_predictions = probability_model.predict(kg_interpolations)
np.save(KG_save_path, kg_predictions)

# ---------------------------------------------------------------
#  GO PREDICTIONS (ROW4)
# ---------------------------------------------------------------

go_interpolations = load_fsdd(GO_path)
go_interpolations = np.squeeze(go_interpolations)
go_interpolations = np.expand_dims(go_interpolations, axis=-1)

go_predictions = probability_model.predict(go_interpolations)
np.save(GO_save_path, go_predictions)

# ---------------------------------------------------------------
#  SO PREDICTIONS (COLUMN0)
# ---------------------------------------------------------------

so_interpolations = load_fsdd(SO_path)
so_interpolations = np.squeeze(so_interpolations)
so_interpolations = np.expand_dims(so_interpolations, axis=-1)

so_predictions = probability_model.predict(so_interpolations)
np.save(SO_save_path, so_predictions)

print('debug')




