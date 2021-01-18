import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def create_simple_generator(input_x, discriminator_output_map_shape=(8,8,256), dropout_rate=0.5):
    alfa_relu=0.2


    x = tf.keras.layers.Dense(discriminator_output_map_shape[0] *
    discriminator_output_map_shape[1] * discriminator_output_map_shape[2])(input_x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape(discriminator_output_map_shape)(x)

    x = tf.keras.layers.Conv2D(256, 5, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)


    x = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(512, 5, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(512, 5, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(3, 7, activation='tanh', padding='same')(x)

    generator = tf.keras.Model(input_x, x)
    generator.summary()
    return generator



def create_simple_discriminator(x_input, dropout_rate=0.5):
    alfa_relu=0.2

    x = tf.keras.layers.Conv2D(256, 5, padding='same')(x_input)
    #x= tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x= tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(1, activation='sigmoid', name='output_fake_real')(x)
    model = tf.keras.Model(x_input, x, name='discriminator')
    return model


def create_simple_adversarial_network(generator, discriminator, latent_space_input):
    generator_output=generator([latent_space_input])
    discriminator_outputs=discriminator(generator_output)
    model=tf.keras.Model(inputs=[latent_space_input], outputs=discriminator_outputs)
    return model

if __name__ == "__main__":
    disc=create_simple_discriminator(x_input=tf.keras.Input(shape=(64,64,3)))
    gen=create_simple_generator(input_x=tf.keras.Input(shape=(128,)), discriminator_output_map_shape=(2,2,256), dropout_rate=0.2)
    create_simple_adversarial_network(gen, disc, tf.keras.Input(shape=(128,)))