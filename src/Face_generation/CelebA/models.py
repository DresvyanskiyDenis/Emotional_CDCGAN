import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def create_simple_generator(input_x, discriminator_output_map_shape=(4,4,256), dropout_rate=0.2):
    alfa_relu=0.2

    x = tf.keras.layers.Dense(discriminator_output_map_shape[0] *
                              discriminator_output_map_shape[1] *
                              discriminator_output_map_shape[2], activation=None)(input_x)
    x = tf.keras.layers.Reshape(discriminator_output_map_shape)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, activation='tanh', strides=(1,1), padding='same')(x)

    model = tf.keras.Model(inputs=[input_x], outputs=x)
    return model



def create_simple_discriminator(x_input, dropout_rate=0.2):
    alfa_relu=0.2

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, use_bias=False, activation=None, strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)
    output_1=tf.keras.layers.Dense(1, activation='sigmoid', name='output_fake_real')(x)

    model = tf.keras.Model(inputs=x_input, outputs=[output_1], name='discriminator')
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