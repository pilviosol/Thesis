import numpy as np
import tensorflow as tf
import pix2pix_modified
import pathlib
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2
import librosa
import librosa.display

AUTOTUNE = tf.data.AUTOTUNE
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

'''
IMPORTANTE: size dell'input e dell'output 
PROVA con CQT che è più semplice e piccola
LEGGERE PAPERS E VEDERE COME SETTANO LORO

'''


# INPUT PIPELINE
train_vn = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_train_256')
train_fl = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_fl_train_256')
test_vn = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_vn_test_256')
test_fl = pathlib.Path('/nas/home/spol/Thesis/URPM_vn_fl/features_fl_test_256')



train_vn_dir = train_vn.iterdir()
test_vn_dir = test_vn.iterdir()
train_fl_dir = train_fl.iterdir()
test_fl_dir = test_fl.iterdir()

train_vn_stft = []
test_vn_stft = []
train_fl_stft = []
test_fl_stft = []

# , STFT: 1025 x length


for idx, feature in enumerate(train_vn_dir):
    feature_name = feature.name
    if "STFTMAG." in feature_name:
        feature_np = np.load(feature)
        feature_reshaped = feature_np[0:1024,0:256]
        print("feature.name: ", feature_name)
        print("feature.shape: ", feature_reshaped.shape)
        train_vn_stft.append(feature_reshaped)


for idx, feature in enumerate(train_fl_dir):
    feature_name = feature.name
    if "STFTMAG." in feature_name:
        feature_np = np.load(feature)
        feature_reshaped = feature_np[0:1024, 0:256]
        print("feature.name: ", feature_name)
        print("feature.shape: ", feature_reshaped.shape)
        train_fl_stft.append(feature_reshaped)


#PRENDO UN SAMPLE PER vn E UNO PER fl E PLOTTO

sample_vn = train_vn_stft[0]
sample_fl = train_fl_stft[3]


print('sample_vn.shape: ', sample_vn.shape)

print('sample_vfl.shape: ', sample_fl.shape)


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(sample_vn, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('vn Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(sample_fl, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('fl Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()




# IMPORTO E UTILIZZO ARCHITETTURA PIX2PIX

OUTPUT_CHANNELS = 1

generator_g = pix2pix_modified.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix_modified.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix_modified.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix_modified.discriminator(norm_type='instancenorm', target=False)

sample_vn = tf.expand_dims(sample_vn, axis=0, name=None)
sample_vn = tf.expand_dims(sample_vn, axis=-1, name=None)
sample_fl = tf.expand_dims(sample_fl, axis=0, name=None)
sample_fl = tf.expand_dims(sample_fl, axis=-1, name=None)


to_fl = generator_g(sample_vn)
print("to_fl.shape", to_fl.shape)
to_vn = generator_f(sample_fl)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_vn, to_fl, sample_fl, to_vn]
title = ['vn', 'To fl', 'fl', 'To vn']

sample_vn_squeezed = tf.squeeze(sample_vn)
to_fl_squeezed = tf.squeeze(to_fl)
sample_fl_squeezed = tf.squeeze(sample_fl)
to_vn_squeezed = tf.squeeze(to_vn)


# DA RIVEDERE, I PLOT FANNO SCHIFO

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False)
img_sample_vn_squeezed = librosa.display.specshow(librosa.amplitude_to_db(sample_vn_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
ax[0].set_title('sample_vn_squeezed')

img_to_fl_squeezed = librosa.display.specshow(librosa.amplitude_to_db(to_fl_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
ax[1].set_title('to_fl_squeezed')

img_sample_fl_squeezed = librosa.display.specshow(librosa.amplitude_to_db(sample_fl_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[2])
ax[2].set_title('sample_fl_squeezed')

img_to_vn_squeezed = librosa.display.specshow(librosa.amplitude_to_db(to_vn_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[3])
ax[3].set_title('to_vn_squeezed')
plt.show()


imgs_squeezed = [img_sample_vn_squeezed, img_to_fl_squeezed, img_sample_fl_squeezed, img_to_vn_squeezed]



plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real fl spectrogram?')
plt.imshow(discriminator_y(sample_fl)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real vn spectrogram?')
plt.imshow(discriminator_x(sample_vn)[0, ..., -1], cmap='RdBu_r')

plt.show()









# LOSS FUNCTIONS

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)








# CHECKPOINTS

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

'''
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
'''



# TRAINING

EPOCHS = 40


def generate_images(model, test_input):
    prediction = model(test_input)
    test_input_squeezed = tf.squeeze(test_input)
    prediction_squeezed = tf.squeeze(prediction)
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        # qui c'era un commento, vedere cycleGAN_forGTZAN_fetaures

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)
        img_sample_vn_squeezed = librosa.display.specshow(librosa.amplitude_to_db(test_input_squeezed, ref=np.max),
                                                             y_axis='log', x_axis='time', ax=ax[0])
        ax[0].set_title('Input Image')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        img_to_fl_squeezed = librosa.display.specshow(librosa.amplitude_to_db(prediction_squeezed, ref=np.max),
                                                         y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set_title('Predicted Image')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.show()


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))


for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in zip(train_vn_stft, train_fl_stft):
        image_x = tf.expand_dims(image_x, axis=0, name=None)
        image_x = tf.expand_dims(image_x, axis=-1, name=None)
        image_y = tf.expand_dims(image_y, axis=0, name=None)
        image_y = tf.expand_dims(image_y, axis=-1, name=None)
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images(generator_g, sample_vn)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))


'''

# qui c'era un commento, vedere cycleGAN_forGTZAN_fetaures
'''
