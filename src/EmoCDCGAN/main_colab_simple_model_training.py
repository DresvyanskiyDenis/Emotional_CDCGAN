import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.EmoCDCGAN.acgan import ACGAN

from src.EmoCDCGAN.utils.data_preprocessing.data_packing import unpack_data_and_labels_npy
from src.EmoCDCGAN.utils.data_preprocessing.preprocess_utils import preprocess_batch_images
from src.EmoCDCGAN.utils.vizualization_utils import visualize_images

def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))

def train():
    path_to_data='D:\\Databases\\AffectNet\\AffectNet\\Batches'
    path_to_save_models='saved_models'
    if not os.path.exists(path_to_save_models):
        os.mkdir(path_to_save_models)

    # params
    latent_space_shape=100
    num_classes=7
    image_size=(128, 128, 3)
    batch_size=int(64)
    mini_batch_size=64
    train_steps=40000
    validate_each_step=100

    # crate class ACGAN
    acgan=ACGAN(latent_space_shape=latent_space_shape, num_classes=num_classes, image_size=image_size)

    # data for validation generator
    noise_validation=np.random.normal(size=(20, latent_space_shape))
    indexes_to_choose = np.random.choice(num_classes, 20)
    labels_validation = np.eye(num_classes)[indexes_to_choose][..., np.newaxis]

    # get list of baches filenames
    batches_filenames=os.listdir(path_to_data)
    batches_filenames=[item.split('.')[0] for item in batches_filenames]
    batches_filenames=np.array(list(set(batches_filenames)))

    # generator model
    generator_model=acgan.create_generator(dropout_rate=0.2)
    #generator_model.load_weights('saved_models/generator.h5')

    # discriminator model
    discriminator_model=acgan.create_discriminator(dropout_rate=0.2)
    optimizer_disc = tf.optimizers.RMSprop(learning_rate=0.0002, decay=6e-8)
    #discriminator_model.load_weights('saved_models/discriminator.h5')
    discriminator_model.compile(optimizer=optimizer_disc, loss={'output_fake_real':'binary_crossentropy',
                                                                'output_class_num':'categorical_crossentropy'},
                                loss_weights={'output_fake_real': 1,
                                              'output_class_num': 1},
                                metrics={'output_fake_real':binary_accuracy})
    discriminator_model.summary()

    # adversarial model
    adversarial_model=acgan.create_adversarial_network(generator_model, discriminator_model)
    optimizer_adv=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=3e-8)
    adversarial_model.compile(optimizer=optimizer_adv, loss={'discriminator':'binary_crossentropy',
                                                            'discriminator_1':'categorical_crossentropy'},
                              metrics={'discriminator':binary_accuracy})

    # summaries
    acgan.print_summaries()
    acgan.create_model_images(path_to_save_models)

    # train process
    for train_step in range(0,train_steps,1):
        # train discriminator
        # take from real data batch_size random images
        rand_num=np.random.randint(0, batches_filenames.shape[0])
        real_images, real_labels= unpack_data_and_labels_npy(path_to_folder=path_to_data,
                                                             filename=batches_filenames[rand_num])
        # preprocessing of real labels/images
        real_images=preprocess_batch_images(real_images, scale=True, resize=True, images_shape=(image_size,image_size,3), bgr=False)
        real_labels=real_labels.expression.values[indexes_to_choose]
        real_labels=tf.keras.utils.to_categorical(real_labels, num_classes=num_classes)


        discriminator_loss, discriminator_acc = acgan.train_discriminator_one_step(batch_size=batch_size,
                                                                                   mini_batch_size= mini_batch_size,
                                                                                   real_images=real_images,
                                                                                   real_labels=real_labels)
        # train generator
        adversarial_loss, adversarial_acc = acgan.train_generator_one_step(batch_size=batch_size*2,
                                       mini_batch_size=mini_batch_size)

        # print the losses
        print('i:%i, Discriminator loss:%f, acc:%f, adversarial loss:%f, acc:%f' % (train_step, discriminator_loss, discriminator_acc,
                                                                                    adversarial_loss, adversarial_acc))

        if train_step % validate_each_step == 0:
            generated_images= generator_model.predict([noise_validation, labels_validation], batch_size=1)
            visualize_images(images=generated_images, labels=labels_validation.squeeze().argmax(axis=-1), path_to_save='images', save_name='generated_images_step_%i.png'%train_step)
            acgan.save_weights_models(path_to_save_models)




if __name__ == "__main__":
    train()
