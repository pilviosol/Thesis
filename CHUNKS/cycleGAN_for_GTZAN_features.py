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
'''for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True) '''

'''
IMPORTANTE: size dell'input e dell'output 
PROVA con CQT che è più semplice e piccola
LEGGERE PAPERS E VEDERE COME SETTANO LORO

'''


# INPUT PIPELINE

train_blues = pathlib.Path('/nas/home/spol/Thesis/GTZAN/features/blues/train/')
train_metal = pathlib.Path('/nas/home/spol/Thesis/GTZAN/features/metal/train/')
test_blues = pathlib.Path('/nas/home/spol/Thesis/GTZAN/features/blues/test/')
test_metal = pathlib.Path('/nas/home/spol/Thesis/GTZAN/features/metal/test/')


train_blues_dir = train_blues.iterdir()
test_blues_dir = test_blues.iterdir()
train_metal_dir = train_metal.iterdir()
test_metal_dir = test_metal.iterdir()

train_blues_cqt = []
train_blues_stftmag = []
train_blues_stftphase = []
test_blues_stftmag = []
test_blues_stftphase = []
test_blues_cqt = []
train_metal_cqt = []
train_metal_stftmag = []
train_metal_stftphase = []
test_metal_stftmag = []
test_metal_stftphase = []
test_metal_cqt = []

# CQT: 84x1293, STFT: 1025x1293


for idx, feature in enumerate(train_blues_dir):
    feature_name = feature.name
    feature_np = np.load(feature)
    feature_reshaped = feature_np[0:1024,0:1024]
    #print("feature.shape: ", feature.shape)
    if "CQT" in str(feature_name):
        train_blues_cqt.append(feature_reshaped)
    elif "STFTMAG" in str(feature_name):
        train_blues_stftmag.append(feature_reshaped)
    else:
        train_blues_stftphase.append(feature_reshaped)


for idx, feature in enumerate(train_metal_dir):
    #print(idx, "    :", feature)
    feature_name = feature.name
    feature_np = np.load(feature)
    feature_reshaped = feature_np[0:1024, 0:1024]
    #print("feature.shape: ", feature.shape)
    if "CQT" in str(feature_name):
        train_metal_cqt.append(feature_reshaped)
    elif "STFTMAG" in str(feature_name):
        train_metal_stftmag.append(feature_reshaped)
    else:
        train_metal_stftphase.append(feature_reshaped)


# FUNZIONI PER CROPPARE E JITTERARE LE IMMAGINI
'''


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256



def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 640 x 480 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = normalize(image)
    return image



train_blues_images = []
train_blues_images_processed = []
train_metal_images = []
train_metal_images_processed = []
test_blues_images = []
test_blues_images_processed = []
test_metal_images = []
test_metal_images_processed = []
'''

# LEGGO E PROCESSO OGNI IMMAGINE, APPENDENDOLE NEGLI ARRAY DEFINITI SOPRA
'''
train_blues_dir = train_blues.iterdir()
for image in train_blues_dir:
    if image.is_file():
        image = cv2.imread(str(image))
        train_blues_images.append(image)
        image = preprocess_image_train(image)
        train_blues_images_processed.append(image)
    else:
        print('Not an image')

train_metal_dir = train_metal.iterdir()
for image in train_metal_dir:
    if image.is_file():
        image = cv2.imread(str(image))
        train_metal_images.append(image)
        image = preprocess_image_train(image)
        train_metal_images_processed.append(image)
    else:
        print('Not an image')

test_blues_dir = test_blues.iterdir()
for image in test_blues_dir:
    if image.is_file():
        image = cv2.imread(str(image))
        test_blues_images.append(image)
        image = preprocess_image_test(image)
        test_blues_images_processed.append(image)
    else:
        print('Not an image')

test_metal_dir = test_metal.iterdir()
for image in test_metal_dir:
    if image.is_file():
        image = cv2.imread(str(image))
        test_metal_images.append(image)
        image = preprocess_image_test(image)
        test_metal_images_processed.append(image)
    else:
        print('Not an image')
'''




#PRENDO UN SAMPLE PER BLUES E UNO PER METAL E PLOTTO

sample_blues = train_blues_stftmag[0]
sample_metal = train_metal_stftmag[0]

audio, sample_rate = librosa.load('../GTZAN/genres_original/blues/blues.00000.wav', res_type='kaiser_fast', mono=True)
cqt = librosa.cqt(y=audio, sr=sample_rate)
stft_mag = np.abs(librosa.stft(y=audio))
stft_phase = np.angle(librosa.stft(y=audio))


plt.figure()
plt.subplot(2,1,1)
plt.imshow(20*np.log10(stft_mag + 1E-5), aspect='auto')

plt.subplot(2,1,2)
plt.imshow(np.unwrap(stft_phase), aspect='auto')
plt.show()

fig_stftmag, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(stft_mag, ref=np.max),
                                           sr=sample_rate, ax=ax)

fig_stftphase, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(stft_phase, ref=np.max),
                                           sr=sample_rate, ax=ax)


print('sample_blues.shape: ', sample_blues.shape)


fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(sample_blues, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Blues Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(sample_metal, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Metal Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


'''
plt.subplot(121)
plt.title('Blues')
plt.imshow(sample_blues * 0.5 + 0.5)
plt.show()

plt.subplot(121)
plt.title('Metal')
plt.imshow(sample_metal * 0.5 + 0.5)
plt.show()
'''



# IMPORTO E UTILIZZO ARCHITETTURA PIX2PIX

OUTPUT_CHANNELS = 1

generator_g = pix2pix_modified.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix_modified.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix_modified.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix_modified.discriminator(norm_type='instancenorm', target=False)

sample_blues = tf.expand_dims(sample_blues, axis=0, name=None)
sample_blues = tf.expand_dims(sample_blues, axis=-1, name=None)
sample_metal = tf.expand_dims(sample_metal, axis=0, name=None)
sample_metal = tf.expand_dims(sample_metal, axis=-1, name=None)


to_metal = generator_g(sample_blues)
print("to_metal.shape", to_metal.shape)
to_blues = generator_f(sample_metal)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_blues, to_metal, sample_metal, to_blues]
title = ['Blues', 'To Metal', 'Metal', 'To Blues']

sample_blues_squeezed = tf.squeeze(sample_blues)
to_metal_squeezed = tf.squeeze(to_metal)
sample_metal_squeezed = tf.squeeze(sample_metal)
to_blues_squeezed = tf.squeeze(to_blues)


# DA RIVEDERE, I PLOT FANNO SCHIFO

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False)
img_sample_blues_squeezed = librosa.display.specshow(librosa.amplitude_to_db(sample_blues_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
ax[0].set_title('sample_blues_squeezed')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

img_to_metal_squeezed = librosa.display.specshow(librosa.amplitude_to_db(to_metal_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
ax[1].set_title('to_metal_squeezed')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

img_sample_metal_squeezed = librosa.display.specshow(librosa.amplitude_to_db(sample_metal_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[2])
ax[2].set_title('sample_metal_squeezed')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

img_to_blues_squeezed = librosa.display.specshow(librosa.amplitude_to_db(to_blues_squeezed, ref=np.max), y_axis='log', x_axis='time', ax=ax[3])
ax[3].set_title('to_blues_squeezed')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


imgs_squeezed = [img_sample_blues_squeezed, img_to_metal_squeezed, img_sample_metal_squeezed, img_to_blues_squeezed]



plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real metal spectrogram?')
plt.imshow(discriminator_y(sample_metal)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real blues spectrogram?')
plt.imshow(discriminator_x(sample_blues)[0, ..., -1], cmap='RdBu_r')

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
        '''
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        #plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.imshow(display_list[i])
        plt.axis('off')
        '''
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)
        img_sample_blues_squeezed = librosa.display.specshow(librosa.amplitude_to_db(test_input_squeezed, ref=np.max),
                                                             y_axis='log', x_axis='time', ax=ax[0])
        ax[0].set_title('Input Image')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        img_to_metal_squeezed = librosa.display.specshow(librosa.amplitude_to_db(prediction_squeezed, ref=np.max),
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
    for image_x, image_y in zip(train_blues_stftmag, train_metal_stftmag):
        image_x = tf.expand_dims(image_x, axis=0, name=None)
        image_x = tf.expand_dims(image_x, axis=-1, name=None)
        image_y = tf.expand_dims(image_y, axis=0, name=None)
        image_y = tf.expand_dims(image_y, axis=-1, name=None)
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the VV_model
    # is clearly visible.
    generate_images(generator_g, sample_blues)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))












'''
# GENERATE USING TEST DATASET

# Run the trained VV_model on the test dataset
# for inp in test_horses_dir.take(5):
test = cv2.imread('/nas/home/spol/Thesis/GTZAN/images/blues/test/blues.00048_CQT.png')
test = tf.expand_dims(
    test, axis=0, name=None
)
generate_images(generator_g, test) '''
