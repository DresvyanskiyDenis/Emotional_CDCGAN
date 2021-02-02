import os

import tensorflow as tf
import numpy as np

from src.PGGAN.utils.data_preprocessing.data_packing import unpack_data_and_labels_npy
from src.PGGAN.utils.data_preprocessing.preprocess_utils import preprocess_batch_images
from src.PGGAN.utils.model_utils import update_fadein, define_generator, define_discriminator, define_composite
from src.PGGAN.utils.vizualization_utils import visualize_images





class PGGAN():
    generators:list
    discriminators:list
    adversarials:list
    latent_space_shape:int
    first_image_size:tuple
    last_image_size:tuple
    num_classes:int=7
    num_blocks:int

    def __init__(self, latent_space_shape:int, first_image_size:tuple, last_image_size:tuple):
        self.latent_space_shape = latent_space_shape
        self.first_image_size = first_image_size
        self.last_image_size = last_image_size
        self.num_blocks = last_image_size[0]//first_image_size[0]

    def load_data_batch(self, path_to_data:str, needed_shape:tuple):
        batches_filenames = np.array(os.listdir(path_to_data))
        batches_filenames = [item.split('.')[0] for item in batches_filenames]
        batches_filenames = np.array(list(set(batches_filenames)))

        rand_num = np.random.randint(0, batches_filenames.shape[0])
        real_images, real_labels = unpack_data_and_labels_npy(path_to_folder=path_to_data,
                                                              filename=batches_filenames[rand_num])
        # preprocessing of real labels/images
        real_images = preprocess_batch_images(real_images, scale=True, resize=True, images_shape=needed_shape, bgr=False)
        real_labels = real_labels.expression.values
        real_labels = tf.keras.utils.to_categorical(real_labels, num_classes=self.num_classes)
        real_binary_labels = np.ones((real_images.shape[0],1))
        return [real_images, real_labels, real_binary_labels]

    def generate_fake_batch(self, generator, batch_size:int):
        # generating fake images
        z = np.random.normal(size=(int(batch_size), self.latent_space_shape))
        fake_images=generator.predict(z)
        # generating fake class labels
        indexes_to_choose = np.random.choice(self.num_classes, batch_size)
        fake_class_labels = np.eye(self.num_classes)[indexes_to_choose]
        # generating binary fake_real labels (-1)
        fake_real_labels = -np.ones((batch_size,1))

        return [z, fake_images, fake_class_labels, fake_real_labels]



    def train_step(self, generator:tf.keras.Model, discriminator:tf.keras.Model, adversarial:tf.keras.Model, path_to_data:str, needed_shape:tuple):
        # train discriminator
        real_images, real_class_labels, real_binary_labels=self.load_data_batch(path_to_data, needed_shape)
        _, fake_images, fake_class_labels, fake_binary_labels=self.generate_fake_batch(generator, batch_size=real_images.shape[0])
        d_loss_real_images, d_acc_real_images=discriminator.train_on_batch(real_images, real_binary_labels)
        d_loss_fake_images, d_acc_fake_images=discriminator.train_on_batch(fake_images, fake_binary_labels)
        # train generator
        latent_points, fake_images, fake_class_labels, fake_binary_labels = self.generate_fake_batch(generator,
                                                                                      batch_size=real_images.shape[0]*2)
        fake_binary_labels=-1.*fake_binary_labels
        adv_batch_size=real_images.shape[0]
        adv_loss, adv_acc = 0., 0.
        for i in range(0, adv_batch_size*2, adv_batch_size):
            tmp_list=adversarial.train_on_batch(latent_points[i:(i+adv_batch_size)], fake_binary_labels[i:(i+adv_batch_size)])
            adv_loss+=tmp_list[0]
            adv_acc += tmp_list[1]
        adv_loss/=2.
        adv_acc /= 2.

        return [d_loss_real_images, d_acc_real_images, d_loss_fake_images, d_acc_fake_images, adv_loss, adv_acc]

    def n_train_steps(self, n_steps:int, generator:tf.keras.Model, discriminator:tf.keras.Model, adversarial:tf.keras.Model, path_to_data:str, needed_shape:tuple, fade_in:bool ):
        for i in range(n_steps):
            if fade_in:
                update_fadein([generator, discriminator, adversarial], i, n_steps)
            loss_acc_list=self.train_step(generator, discriminator, adversarial, path_to_data, needed_shape)
            print('discriminator real images loss:%f, acc:%f, discriminator fake images loss:%f, acc:%f, generator loss:%f, acc:%f'
                  %(loss_acc_list[0],loss_acc_list[1],loss_acc_list[2],loss_acc_list[3],loss_acc_list[4],loss_acc_list[5]))

    def create_generators(self):
        self.generators = define_generator(self.latent_space_shape, self.num_blocks, in_dim=self.first_image_size[0])

    def create_discriminators(self):
        self.discriminators = define_discriminator(n_blocks=self.num_blocks, input_shape=self.first_image_size)

    def create_advesarials(self):
        self.adversarials = define_composite(self.discriminators, self.generators)

    def visualize_generator(self, generator:tf.keras.Model, latent_space_data: np.ndarray, path_to_save, specified_name=''):
        generated_images=generator.predict(latent_space_data)
        visualize_images(generated_images, labels=np.array([0 for i in range(generated_images.shape[0])]),
                         path_to_save=path_to_save, save_name=specified_name)



    def train_process(self, n_steps_per_block:list, n_batch_per_block:list, path_to_data:str):
        # validation points to visualise
        validation_latent_points= np.random.normal(size=(int(20), self.latent_space_shape))
        indexes_to_choose = np.random.choice(self.num_classes, 20)
        fake_class_labels = np.eye(self.num_classes)[indexes_to_choose]
        # start to train model
        g_normal, d_normal, adv_normal = self.generators[0][0], self.discriminators[0][0], self.adversarials[0][0]
        self.n_train_steps(n_steps=n_steps_per_block[0],generator=g_normal, discriminator=d_normal, adversarial=adv_normal,
                           path_to_data=path_to_data, needed_shape=self.first_image_size, fade_in=False)
        self.visualize_generator(g_normal, validation_latent_points, path_to_save='generated_images', specified_name='pretrain.png')

        # start cycle of growing and training
        current_image_shape=self.first_image_size
        for i in range(1, len(self.generators)):
            g_normal, g_fadein = self.generators[i]
            d_normal, d_fadein = self.discriminators[i]
            adv_normal, adv_fadein = self.adversarials[i]
            # update image shape
            current_image_shape=(current_image_shape[0]*2, current_image_shape[1]*2, 3)
            # train with fade_in
            self.n_train_steps(n_steps=n_steps_per_block[i],generator=g_fadein, discriminator=d_fadein, adversarial=adv_fadein,
                           path_to_data=path_to_data, needed_shape=current_image_shape, fade_in=True)
            self.visualize_generator(g_fadein, validation_latent_points, path_to_save='generated_images',
                                     specified_name='faded_block_%i.png'%(i))
            # train normal models
            self.n_train_steps(n_steps=n_steps_per_block[i],generator=g_normal, discriminator=d_normal, adversarial=adv_normal,
                           path_to_data=path_to_data, needed_shape=current_image_shape, fade_in=False)
            self.visualize_generator(g_normal, validation_latent_points, path_to_save='generated_images',
                                     specified_name='normal_block_%i.png' % (i))

