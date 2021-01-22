import tensorflow as tf
import os
import gc
from numpy.core.multiarray import ndarray

from src.EmoCDCGAN.models import create_simple_generator, create_simple_discriminator, PixelNormLayer
import numpy as np

from src.EmoCDCGAN.utils.data_preprocessing.preprocess_utils import add_noise_in_labels

def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))


class ACGAN():
    generator:tf.keras.Model
    discriminator:tf.keras.Model
    adversarial:tf.keras.Model
    latent_space_shape:int
    num_classes:int
    image_size:tuple


    def load_models(self,path_to_models:str):
        self.discriminator=tf.keras.models.load_model(os.path.join(path_to_models,'discriminator.h5'), custom_objects={'PixelNormLayer': PixelNormLayer})
        self.generator=tf.keras.models.load_model(os.path.join(path_to_models,'generator.h5'), custom_objects={'PixelNormLayer': PixelNormLayer},compile=False)
        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss={'output_fake_real': 'binary_crossentropy',
                                                                    'output_class_num': 'categorical_crossentropy'},
                                    loss_weights={'output_fake_real': 1,
                                                  'output_class_num': 1},
                                   metrics={'output_fake_real': binary_accuracy})
        adv_tmp=tf.keras.models.load_model(os.path.join(path_to_models,'adversarial.h5'), custom_objects={'PixelNormLayer': PixelNormLayer})
        adv_opt=adv_tmp.optimizer
        self.adversarial=self.create_adversarial_network(self.generator, self.discriminator)
        self.adversarial.compile(optimizer=adv_opt, loss={'discriminator': 'binary_crossentropy',
                                        'discriminator_1': 'categorical_crossentropy'},
                                  metrics={'discriminator': binary_accuracy})
        del adv_tmp
        gc.collect()
        return self.generator, self.discriminator, self.adversarial


    def __init__(self, latent_space_shape:int, num_classes:int, image_size:tuple):
        self.latent_space_shape=latent_space_shape
        self.num_classes=num_classes
        self.image_size=image_size

    def create_generator(self, dropout_rate:float=0.2):
        z_input_shape=(self.latent_space_shape,)
        c_input_shape=(self.num_classes,)
        input_x_gen = tf.keras.layers.Input(z_input_shape)
        input_y = tf.keras.layers.Input(c_input_shape)
        generator_model = create_simple_generator(input_x_gen, input_y, dropout_rate=dropout_rate)
        self.generator=generator_model
        return generator_model

    def create_discriminator(self, dropout_rate:float=0.2):
        x_input_shape=self.image_size
        input_x_disc = tf.keras.layers.Input(x_input_shape)
        discriminator_model = create_simple_discriminator(x_input=input_x_disc, num_classes=self.num_classes,
                                                          dropout_rate=dropout_rate)
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

    def save_models(self, output_path:str=''):
        self.generator.save(os.path.join(output_path,'generator.h5'))
        self.discriminator.save(os.path.join(output_path,'discriminator.h5'))
        self.adversarial.save(os.path.join(output_path,'adversarial.h5'))

    def train_discriminator_one_step(self, batch_size:int, mini_batch_size:int, real_images:ndarray, real_labels:ndarray):
        # train discriminator
        # generate random images from generator
        z = np.random.normal(size=(int(batch_size), self.latent_space_shape))
        indexes_to_choose = np.random.choice(self.num_classes, int(batch_size))
        fake_labels = np.eye(self.num_classes)[indexes_to_choose]
        fake_images = self.generator.predict([z, fake_labels], batch_size=1)

        # add noise for real images
        real_images = real_images + 0.05*np.random.normal(size=real_images.shape)

        # concatenate
        train_discriminator_batch_images = np.concatenate([fake_images, real_images], axis=0).astype('float32')
        train_discriminator_labels_images = np.concatenate([fake_labels, real_labels], axis=0)
        y_discriminator = np.ones((batch_size * 2,))
        y_discriminator[:batch_size] = 0
        y_discriminator = add_noise_in_labels(y_discriminator)


        permut=np.random.permutation(train_discriminator_batch_images.shape[0])
        train_discriminator_batch_images = train_discriminator_batch_images[permut]
        train_discriminator_labels_images = train_discriminator_labels_images[permut]
        y_discriminator = y_discriminator[permut]

        # train discriminator
        discriminator_loss = 0
        discriminator_acc = 0
        for batch_step in range(train_discriminator_batch_images.shape[0] // mini_batch_size):
            start = batch_step * mini_batch_size
            end = (batch_step + 1) * mini_batch_size
            loss = self.discriminator.train_on_batch(x=train_discriminator_batch_images.astype('float32')[start:end],
                                                      y=[y_discriminator[start:end],
                                                         train_discriminator_labels_images[start:end]])
            discriminator_loss += loss[0]
            discriminator_acc += loss[-1]
        discriminator_loss /= float(train_discriminator_batch_images.shape[0] // mini_batch_size)
        discriminator_acc /= float(train_discriminator_batch_images.shape[0] // mini_batch_size)
        #print(self.discriminator.layers[3].weights[0])
        return [discriminator_loss, discriminator_acc]

    def train_generator_one_step(self, batch_size:int, mini_batch_size:int):
        # train generator
        z = np.random.normal(size=(int(batch_size), self.latent_space_shape))
        indexes_to_choose = np.random.choice(self.num_classes, batch_size)
        fake_labels = np.eye(self.num_classes)[indexes_to_choose]
        y_adversarial_network = np.ones((batch_size,))
        y_adversarial_network = add_noise_in_labels(y_adversarial_network)

        # train adversarial model
        adversarial_loss = 0
        adversarial_acc= 0
        for batch_step in range(z.shape[0] // mini_batch_size):
            start = batch_step * mini_batch_size
            end = (batch_step + 1) * mini_batch_size
            loss = self.adversarial.train_on_batch(x=[z.astype('float32')[start:end],
                                                                    fake_labels[start:end]],
                                                                 y=[y_adversarial_network[start:end],
                                                                    fake_labels[start:end]])
            adversarial_loss += loss[0]
            adversarial_acc +=loss[-1]
        adversarial_loss /= float(z.shape[0] // mini_batch_size)
        adversarial_acc /= float(z.shape[0] // mini_batch_size)
        return [adversarial_loss, adversarial_acc]