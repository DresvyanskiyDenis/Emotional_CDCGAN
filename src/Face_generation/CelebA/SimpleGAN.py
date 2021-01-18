import tensorflow as tf
import os

from numpy.core.multiarray import ndarray

import numpy as np

from src.Face_generation.CelebA.models import create_simple_generator, create_simple_discriminator
from src.Face_generation.CelebA.utils.data_preprocessing.preprocess_utils import add_noise_in_labels


class SimpleGAN():
    generator:tf.keras.Model
    discriminator:tf.keras.Model
    adversarial:tf.keras.Model
    latent_space_shape:int
    image_size:tuple



    def __init__(self, latent_space_shape:int, image_size:tuple):
        self.latent_space_shape=latent_space_shape
        self.image_size=image_size

    def create_generator(self, dropout_rate:float=0.2):
        z_input_shape=(self.latent_space_shape,)
        input_x_gen = tf.keras.layers.Input(z_input_shape)
        generator_model = create_simple_generator(input_x_gen, dropout_rate=dropout_rate)
        self.generator=generator_model
        return generator_model

    def create_discriminator(self, dropout_rate:float=0.2):
        x_input_shape=self.image_size
        input_x_disc = tf.keras.layers.Input(x_input_shape)
        discriminator_model = create_simple_discriminator(x_input=input_x_disc, dropout_rate=dropout_rate)
        self.discriminator=discriminator_model
        return discriminator_model

    def create_adversarial_network(self, generator:tf.keras.Model, discriminator:tf.keras.Model):
        self.discriminator.trainable = False
        discriminator_outputs = discriminator(generator.outputs)
        model = tf.keras.Model(inputs=generator.inputs, outputs=discriminator_outputs)
        self.adversarial=model
        return model

    def print_summaries(self):
        self.generator.summary()
        self.discriminator.summary()
        self.adversarial.summary()

    def create_model_images(self, output_path:str=''):
        tf.keras.utils.plot_model(self.generator, show_shapes=True, to_file=os.path.join(output_path,"model_gen.png"))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, to_file=os.path.join(output_path,"model_disc.png"))
        tf.keras.utils.plot_model(self.adversarial, show_shapes=True, to_file=os.path.join(output_path,"model_advers.png"))

    def save_weights_models(self, output_path:str=''):
        self.generator.save_weights(os.path.join(output_path,'generator.h5'))
        self.discriminator.save_weights(os.path.join(output_path,'discriminator.h5'))
        self.adversarial.save_weights(os.path.join(output_path,'adversarial.h5'))

    def train_discriminator_one_step(self, batch_size:int, mini_batch_size:int, real_images:ndarray):
        # train discriminator
        # generate random images from generator
        z = np.random.normal(size=(int(batch_size), self.latent_space_shape))
        fake_images = self.generator.predict([z], batch_size=1)

        # concatenate
        train_discriminator_batch_images = np.concatenate([fake_images, real_images], axis=0).astype('float32')
        y_discriminator = np.ones((batch_size * 2,))
        y_discriminator[:batch_size] = 0
        y_discriminator = add_noise_in_labels(y_discriminator)

        # train discriminator
        discriminator_loss = 0
        discriminator_acc = 0
        for batch_step in range(train_discriminator_batch_images.shape[0] // mini_batch_size):
            start = batch_step * mini_batch_size
            end = (batch_step + 1) * mini_batch_size
            loss = self.discriminator.train_on_batch(x=train_discriminator_batch_images.astype('float32')[start:end],
                                                      y=[y_discriminator[start:end]])
            discriminator_loss += loss[0]
            discriminator_acc += loss[-1]
        discriminator_loss /= float(train_discriminator_batch_images.shape[0] // mini_batch_size)
        discriminator_acc /= float(train_discriminator_batch_images.shape[0] // mini_batch_size)
        return [discriminator_loss, discriminator_acc]

    def train_generator_one_step(self, batch_size:int, mini_batch_size:int):
        # train generator
        z = np.random.normal(size=(int(batch_size), self.latent_space_shape))
        y_adversarial_network = np.ones((batch_size,))
        #y_adversarial_network = add_noise_in_labels(y_adversarial_network)

        # train adversarial model
        adversarial_loss = 0
        adversarial_acc= 0
        for batch_step in range(z.shape[0] // mini_batch_size):
            start = batch_step * mini_batch_size
            end = (batch_step + 1) * mini_batch_size
            loss = self.adversarial.train_on_batch(x=[z.astype('float32')[start:end]],
                                                                 y=[y_adversarial_network[start:end]])
            adversarial_loss += loss[0]
            adversarial_acc +=loss[-1]
        adversarial_loss /= float(z.shape[0] // mini_batch_size)
        adversarial_acc /= float(z.shape[0] // mini_batch_size)
        return [adversarial_loss, adversarial_acc]