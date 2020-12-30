import os
import numpy as np
import tensorflow as tf

from src.EmoCDCGAN.utils.data_preprocessing.load_utils import load_AffectNet_labels, load_image, crop_image, preprocess_image
from src.EmoCDCGAN.resnet_blocks import create_generator_resnet_based, create_discriminator_resnet_based, \
    build_adversarial_model_resnet_based
from src.EmoCDCGAN.utils.train_utils import train_n_mini_batches
from src.EmoCDCGAN.utils.vizualization_utils import visualize_images


def train():
    path_to_images='E:\\Databases\\AffectNet\\AffectNet\\zip\\Manually_Annotated_Images'
    path_to_labels='E:\\Databases\\AffectNet\\AffectNet\\zip\\training.csv'
    labels=load_AffectNet_labels(path_to_labels)


    # params
    latent_space_shape=200
    num_classes=7
    image_size=224
    batch_size=int(32)
    train_steps=40000
    validate_each_step=50

    # data for validation generator
    noise_validation=np.random.uniform(-1., 1., (20, latent_space_shape,1))
    indexes_to_choose = np.random.choice(num_classes, 20)
    labels_validation = np.eye(num_classes)[indexes_to_choose][..., np.newaxis]

    # generator model
    input_x_gen=tf.keras.layers.Input((latent_space_shape,))
    input_y=tf.keras.layers.Input((num_classes,))
    generator_model=create_generator_resnet_based(input_x_gen, input_y)

    # discriminator model
    input_x_disc=tf.keras.layers.Input((image_size, image_size, 3))
    discriminator_model=create_discriminator_resnet_based(input_x_disc, input_y, image_size)
    #optimizer_disc=tf.keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
    optimizer_disc = tf.keras.optimizers.Adam(lr=0.0002, amsgrad=True)
    discriminator_model.compile(optimizer=optimizer_disc, loss='binary_crossentropy')

    # adversarial model
    discriminator_model.trainable=False
    adversarial_model=build_adversarial_model_resnet_based(generator_model, discriminator_model, input_x_gen, input_y)
    #optimizer_adv=tf.keras.optimizers.RMSprop(lr=0.0002*0.5, decay=6e-8*0.5)
    optimizer_adv=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=True)
    adversarial_model.compile(optimizer=optimizer_adv, loss='binary_crossentropy')

    # summaries
    generator_model.summary()
    tf.keras.utils.plot_model(generator_model, show_shapes=True, to_file="model_gen.png")
    discriminator_model.summary()
    tf.keras.utils.plot_model(discriminator_model, show_shapes=True, to_file="model_disc.png")
    adversarial_model.summary()
    tf.keras.utils.plot_model(adversarial_model, show_shapes=True, to_file="model_advers.png")

    # train process
    for train_step in range(train_steps):
        # train discriminator
        # generate random images from generator
        z = np.random.uniform(-1., 1., (int(batch_size ), latent_space_shape))
        indexes_to_choose = np.random.choice(num_classes, int(batch_size))
        fake_labels = np.eye(num_classes)[indexes_to_choose]
        fake_images = generator_model.predict([z, fake_labels], batch_size=1)

        # take from real data batch_size random images
        indexes_to_choose = np.random.choice(labels.shape[0], int(batch_size))
        real_images=np.zeros((batch_size, image_size, image_size, 3))
        for i in range(len(indexes_to_choose)):
            img=load_image(os.path.join(path_to_images, labels.iloc[i, 0]))
            img=crop_image(img, labels.iloc[i,1], labels.iloc[i,2], labels.iloc[i,3], labels.iloc[i,4])
            img=preprocess_image(img, scale=True, resize=True, needed_shape=(224,224,3), bgr=False)
            real_images[i]=img
        real_labels=labels.expression.values[indexes_to_choose]
        real_labels=tf.keras.utils.to_categorical(real_labels, num_classes=num_classes)

        # concatenate
        train_discriminator_batch_images = np.concatenate([fake_images, real_images], axis=0)
        train_discriminator_batch_labels = np.concatenate([fake_labels, real_labels], axis=0)
        y_discriminator = np.ones((batch_size*2,))
        y_discriminator[:batch_size] = 0

        # train discriminator
        descriminator_loss=train_n_mini_batches(model=discriminator_model,
                                                data=[train_discriminator_batch_images, train_discriminator_batch_labels],
                                                labels=y_discriminator,
                                                num_mini_batches=int(y_discriminator.shape[0]/2),
                                                batch_size=2, loss=tf.keras.losses.binary_crossentropy)
        #descriminator_loss=discriminator_model.train_on_batch([train_discriminator_batch_images, train_discriminator_batch_labels],y_discriminator)

        # train generator
        gen_batch_size = batch_size
        z = np.random.uniform(-1., 1., (gen_batch_size, latent_space_shape))
        indexes_to_choose = np.random.choice(num_classes, gen_batch_size)
        fake_labels = np.eye(num_classes)[indexes_to_choose]
        y_adversarial_network = np.ones((gen_batch_size,))

        # train adversarial model
        adversarial_loss = train_n_mini_batches(model=adversarial_model,
                                                  data=[z,
                                                        fake_labels],
                                                  labels=y_adversarial_network,
                                                  num_mini_batches=int(y_adversarial_network.shape[0] / 2),
                                                  batch_size=2, loss=tf.keras.losses.binary_crossentropy)

        #adversarial_loss = adversarial_model.train_on_batch([z, fake_labels], y_adversarial_network)

        # print the losses
        print('i:%i, Discriminator loss:%f, adversarial loss:%f' % (train_step, descriminator_loss, adversarial_loss))

        if train_step % validate_each_step == 0:
            generated_images= generator_model.predict([noise_validation, labels_validation], batch_size=1)
            visualize_images(images=generated_images, labels=np.argmax(labels_validation.squeeze(), axis=-1), path_to_save='images', save_name='generated_images_step_%i.png'%train_step)

if __name__ == "__main__":
    train()