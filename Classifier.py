import numpy as np
import pathlib
from functions import load_fsdd
import tensorflow as tf

# ---------------------------------------------------------------
#  PATHS, VARIABLES
# ---------------------------------------------------------------
string_train_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_strings/'
keyboard_train_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_keyboards/'
guitar_train_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_guitars/'
organ_train_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_organs/'
string_train_path = pathlib.Path(string_train_path)
keyboard_train_path = pathlib.Path(keyboard_train_path)
guitar_train_path = pathlib.Path(guitar_train_path)
organ_train_path = pathlib.Path(organ_train_path)

string_val_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_string_0305_VALID/'
keyboard_val_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_keyboard_0305_VALID/'
guitar_val_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_guitar_1805_VALID/'
organ_val_path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_organ_1805_VALID/'
string_val_path = pathlib.Path(string_val_path)
keyboard_val_path = pathlib.Path(keyboard_val_path)
guitar_val_path = pathlib.Path(guitar_val_path)
organ_val_path = pathlib.Path(organ_val_path)


classes = [0, 1, 2, 3]

model_path = "/nas/home/spol/Thesis/Classifier.h5"
# ---------------------------------------------------------------
#  IMPORT AND ORGANIZE THE DATA
# ---------------------------------------------------------------

# ------ TRAIN -------
string_train = load_fsdd(string_train_path)
keyboard_train = load_fsdd(keyboard_train_path)
guitar_train = load_fsdd(guitar_train_path)
organ_train = load_fsdd(organ_train_path)

x_train = np.concatenate((string_train, keyboard_train, guitar_train, organ_train), axis=0)

y_train = []
for i in range(2832):
    if i < 708:
        y_train.append(classes[0])
    elif 708 <= i < 1416:
        y_train.append(classes[1])
    elif 1416 <= i < 2132:
        y_train.append(classes[2])
    else:
        y_train.append(classes[3])
y_train = np.asarray(y_train)


# ------ VAL -------
string_val = load_fsdd(string_val_path)
keyboard_val = load_fsdd(keyboard_val_path)
guitar_val = load_fsdd(guitar_val_path)
organ_val = load_fsdd(organ_val_path)

x_val = np.concatenate((string_val, keyboard_val, guitar_val, organ_val), axis=0)

y_val = []
for i in range(352):
    if i < 88:
        y_val.append(classes[0])
    elif 88 <= i < 176:
        y_val.append(classes[1])
    elif 176 <= i < 264:
        y_val.append(classes[2])
    else:
        y_val.append(classes[3])
y_val = np.asarray(y_val)


# ---------------------------------------------------------------
#  DEFINE THE MODEL
# ---------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(512, 64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_val, y_val, verbose=2)

print('\nTest accuracy:', test_acc)


model.save(model_path)




