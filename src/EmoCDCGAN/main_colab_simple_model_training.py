import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.EmoCDCGAN.models import create_simple_generator, create_simple_discriminator, create_simple_adversarial_network
from src.EmoCDCGAN.resnet_blocks import create_generator_resnet_based, create_discriminator_resnet_based, \
    build_adversarial_model_resnet_based
from src.EmoCDCGAN.utils.data_preprocessing.data_packing import unpack_data_and_labels_npy
from src.EmoCDCGAN.utils.data_preprocessing.preprocess_utils import preprocess_image, preprocess_batch_images, \
    shuffle_ndarrays, add_noise_in_labels
from src.EmoCDCGAN.utils.train_utils import train_n_mini_batches
from src.EmoCDCGAN.utils.vizualization_utils import visualize_images


def train():
    path_to_data='D:\\Databases\\AffectNet\\AffectNet\\Batches'

    # params
    latent_space_shape=128
    num_classes=7
    image_size=64
    batch_size=int(64)
    mini_batch_size=16
    train_steps=40000
    validate_each_step=1

    # data for validation generator
    noise_validation=np.random.normal(size=(20, latent_space_shape))
    indexes_to_choose = np.random.choice(num_classes, 20)
    labels_validation = np.eye(num_classes)[indexes_to_choose][..., np.newaxis]

    # get list of baches filenames
    batches_filenames=os.listdir(path_to_data)
    batches_filenames=[item.split('.')[0] for item in batches_filenames]
    batches_filenames=np.array(list(set(batches_filenames)))

    # generator model
    input_x_gen=tf.keras.layers.Input((latent_space_shape,))
    input_y=tf.keras.layers.Input((num_classes,))
    generator_model=create_simple_generator(input_x_gen, input_y)
    #generator_model.load_weights('saved_models/generator.h5')

    # discriminator model
    input_x_disc=tf.keras.layers.Input((image_size, image_size, 3))
    discriminator_model=create_simple_discriminator(x_input=input_x_disc, num_classes=num_classes, dropout_rate=0.2)
    #optimizer_disc=tf.keras.optimizers.RMSprop(lr=0.0001, decay=6e-8)
    optimizer_disc = tfa.optimizers.AdamW(learning_rate=0.0002, weight_decay=10e-6)
    #discriminator_model.load_weights('saved_models/discriminator.h5')
    discriminator_model.compile(optimizer=optimizer_disc, loss={'output_fake_real':'binary_crossentropy',
                                                                'output_class_num':'categorical_crossentropy'},
                                loss_weights={'output_fake_real': 1,
                                              'output_class_num': 1},
                                metrics={'output_fake_real':'acc'})

    # adversarial model
    discriminator_model.trainable=False
    adversarial_model=create_simple_adversarial_network(generator_model, discriminator_model, input_x_gen, input_y)
    #optimizer_adv=tf.keras.optimizers.RMSprop(lr=0.0001, decay=6e-8)
    #optimizer_adv=tf.keras.optimizers.Adam(lr=0.0005, amsgrad=True)
    optimizer_adv=tf.keras.optimizers.SGD(learning_rate=0.001)
    adversarial_model.compile(optimizer=optimizer_adv, loss={'discriminator':'binary_crossentropy',
                                                            'discriminator_1':'categorical_crossentropy'},
                              metrics={'discriminator':'acc'})
    #discriminator_model.trainable = True

    # summaries
    generator_model.summary()
    tf.keras.utils.plot_model(generator_model, show_shapes=True, to_file="model_gen.png")
    discriminator_model.summary()
    tf.keras.utils.plot_model(discriminator_model, show_shapes=True, to_file="model_disc.png")
    adversarial_model.summary()
    tf.keras.utils.plot_model(adversarial_model, show_shapes=True, to_file="model_advers.png")

    # train process
    for train_step in range(0,train_steps,1):
        # train discriminator
        # generate random images from generator
        z = np.random.normal(size=(int(batch_size), latent_space_shape))
        indexes_to_choose = np.random.choice(num_classes, int(batch_size))
        fake_labels = np.eye(num_classes)[indexes_to_choose]
        fake_images = generator_model.predict([z, fake_labels], batch_size=1)

        # take from real data batch_size random images
        rand_num=np.random.randint(0, batches_filenames.shape[0])
        real_images, real_labels= unpack_data_and_labels_npy(path_to_folder=path_to_data,
                                                             filename=batches_filenames[rand_num])
        # preprocessing
        real_images=preprocess_batch_images(real_images, scale=True, resize=True, images_shape=(64,64,3), bgr=False)
        real_labels=real_labels.expression.values[indexes_to_choose]
        real_labels=tf.keras.utils.to_categorical(real_labels, num_classes=num_classes)

        # concatenate
        train_discriminator_batch_images = np.concatenate([fake_images, real_images], axis=0).astype('float32')
        train_discriminator_labels_images=np.concatenate([fake_labels, real_labels], axis=0)
        y_discriminator = np.ones((batch_size*2,))
        y_discriminator[:batch_size] = 0
        y_discriminator=add_noise_in_labels(y_discriminator)

        # shuffle
        #train_discriminator_batch_images, \
        #train_discriminator_batch_labels, \
        #y_discriminator = shuffle_ndarrays([train_discriminator_batch_images, train_discriminator_batch_labels, y_discriminator])

        # train discriminator
        discriminator_loss=0
        for batch_step in range(train_discriminator_batch_images.shape[0]//mini_batch_size):
            start=batch_step*mini_batch_size
            end=(batch_step+1)*mini_batch_size
            discriminator_loss+=discriminator_model.train_on_batch(x=train_discriminator_batch_images.astype('float32')[start:end],
                                               y=[y_discriminator[start:end], train_discriminator_labels_images[start:end]])[0]
        discriminator_loss/=float(train_discriminator_batch_images.shape[0]//mini_batch_size)

        # train generator
        gen_batch_size = batch_size*2
        z = np.random.normal(size=(int(gen_batch_size ), latent_space_shape))
        indexes_to_choose = np.random.choice(num_classes, gen_batch_size)
        fake_labels = np.eye(num_classes)[indexes_to_choose]
        y_adversarial_network = np.ones((gen_batch_size,))
        y_adversarial_network = add_noise_in_labels(y_adversarial_network)

        # train adversarial model
        adversarial_loss=0
        for batch_step in range(z.shape[0]//mini_batch_size):
            start=batch_step*mini_batch_size
            end=(batch_step+1)*mini_batch_size
            adversarial_loss+=adversarial_model.train_on_batch(x=[z.astype('float32')[start:end],
                                                                      fake_labels[start:end]],
                                               y=[y_adversarial_network[start:end], fake_labels[start:end]])[0]
        adversarial_loss/=float(z.shape[0]//mini_batch_size)


        #adversarial_loss = adversarial_model.train_on_batch([z, fake_labels], y_adversarial_network)

        # print the losses
        print('i:%i, Discriminator loss:%f, adversarial loss:%f' % (train_step, discriminator_loss, adversarial_loss))

        if train_step % validate_each_step == 0:
            generated_images= generator_model.predict([noise_validation, labels_validation], batch_size=1)
            visualize_images(images=generated_images, labels=np.argmax(labels_validation.squeeze(), axis=-1), path_to_save='images', save_name='generated_images_step_%i.png'%train_step)
            generator_model.save_weights('generator.h5')
            discriminator_model.save_weights('discriminator.h5')
            adversarial_model.save_weights('adversarial.h5')




if __name__ == "__main__":
    train()
