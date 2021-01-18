import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.Face_generation.CelebA.SimpleGAN import SimpleGAN
from src.Face_generation.CelebA.utils.data_preprocessing.data_packing import unpack_data_npy
from src.Face_generation.CelebA.utils.data_preprocessing.preprocess_utils import preprocess_batch_images
from src.Face_generation.CelebA.utils.vizualization_utils import visualize_images


def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))


def train():
    path_to_data = '/content/drive/MyDrive/Batches_celebA'
    path_to_save_models = 'saved_models'
    if not os.path.exists(path_to_save_models):
        os.mkdir(path_to_save_models)

    # params
    latent_space_shape = 100
    image_size = (128, 128, 3)
    batch_size = int(25)
    mini_batch_size = 50
    train_steps = 40000
    validate_each_step = 50

    # crate class SimpleGAN
    simpleGAN = SimpleGAN(latent_space_shape=latent_space_shape, image_size=image_size)

    # data for validation generator
    noise_validation = np.random.uniform(-1, 1, size=(20, latent_space_shape))

    # get list of baches filenames
    batches_filenames = os.listdir(path_to_data)
    batches_filenames = [item.split('.')[0] for item in batches_filenames]
    batches_filenames = np.array(list(set(batches_filenames)))

    # generator model
    generator_model = simpleGAN.create_generator(dropout_rate=0.0)
    generator_model.load_weights('saved_models/generator.h5')

    # discriminator model
    discriminator_model = simpleGAN.create_discriminator(dropout_rate=0.0)
    optimizer_disc = tf.keras.optimizers.RMSprop(lr=.00001, clipvalue=1.0, decay=1e-8)
    discriminator_model.load_weights('saved_models/discriminator.h5')
    discriminator_model.compile(optimizer=optimizer_disc, loss={'output_fake_real': 'binary_crossentropy'},
                                metrics={'output_fake_real': binary_accuracy})
    discriminator_model.summary()

    # adversarial model
    adversarial_model = simpleGAN.create_adversarial_network(generator_model, discriminator_model)
    optimizer_adv = tf.keras.optimizers.RMSprop(lr=.000025, clipvalue=1.0, decay=1e-8)
    adversarial_model.compile(optimizer=optimizer_adv, loss={'discriminator': 'binary_crossentropy'},
                              metrics={'discriminator': binary_accuracy})

    # summaries
    simpleGAN.print_summaries()
    simpleGAN.create_model_images(path_to_save_models)

    # train process
    for train_step in range(1951, train_steps, 1):
        # train discriminator
        # take from real data batch_size random images
        rand_num = np.random.randint(0, batches_filenames.shape[0])

        real_images = unpack_data_npy(path_to_folder=path_to_data,
                                      filename=batches_filenames[rand_num])
        perm = np.random.permutation(real_images.shape[0])
        perm = perm[:batch_size]
        real_images = real_images[perm]
        # preprocessing of real labels/images
        real_images = preprocess_batch_images(real_images, scale=True, resize=False, images_shape=image_size, bgr=False)

        discriminator_loss, discriminator_acc = simpleGAN.train_discriminator_one_step(batch_size=batch_size,
                                                                                       mini_batch_size=mini_batch_size,
                                                                                       real_images=real_images)
        # train generator
        batch_size_generator = batch_size * 2
        adversarial_loss, adversarial_acc = simpleGAN.train_generator_one_step(batch_size=batch_size_generator,
                                                                               mini_batch_size=mini_batch_size)

        # print the losses
        print('i:%i, Discriminator loss:%f, acc:%f, adversarial loss:%f, acc:%f' % (
        train_step, discriminator_loss, discriminator_acc,
        adversarial_loss, adversarial_acc))

        if train_step % validate_each_step == 0:
            generated_images = generator_model.predict([noise_validation], batch_size=1)
            visualize_images(images=generated_images, labels=[i for i in range(generated_images.shape[0])],
                             path_to_save='images', save_name='generated_images_step_%i.png' % train_step)
            simpleGAN.save_weights_models(path_to_save_models)


if __name__ == "__main__":
    train()
