import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# Pixelwise feature vector normalization.
class PixelNormLayer(Layer):
    def __init__(self,**kwargs):
        super(PixelNormLayer,self).__init__(**kwargs)
    def call(self, inputs, **kwargs):
        return inputs / K.sqrt(K.mean(inputs**2, axis=-1, keepdims=True) + 1.0e-8)
    def compute_output_shape(self, input_shape):
        return input_shape

def create_simple_generator(input_x, input_y, discriminator_output_map_shape=(4,4,256), dropout_rate=0.2):
    alfa_relu=0.1
    group=128
    concat = tf.keras.layers.concatenate([input_x, input_y])
    x = tf.keras.layers.Flatten()(concat)
    x = tf.keras.layers.Dense(discriminator_output_map_shape[0] *
                              discriminator_output_map_shape[1] *
                              discriminator_output_map_shape[2], activation=None)(x)
    x = tf.keras.layers.Reshape(discriminator_output_map_shape)(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    '''x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
   # x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)'''

    x=tf.keras.layers.Conv2D(filters=256, kernel_size=4, use_bias=False, activation=None, padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.Conv2D(filters=512, kernel_size=4, use_bias=False, activation=None, padding='same')(x)
    x = PixelNormLayer()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=3, kernel_size=5, activation='tanh', strides=(1,1), padding='same', name='generator_output')(x)

    model = tf.keras.Model(inputs=[input_x, input_y], outputs=x)
    return model



def create_simple_discriminator(x_input, num_classes, dropout_rate=0.2):
    alfa_relu=0.1

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=64, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same'))(x_input)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=128, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same'))(x_input)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=128, kernel_size=5, use_bias=False, activation=None, strides=(2,2), padding='same'))(x_input)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=256, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same'))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=512, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same'))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)

    '''x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=512, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same'))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)'''

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=512, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same' ))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output_conv = tf.keras.layers.LeakyReLU(alfa_relu)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=1024, kernel_size=4, use_bias=False, activation=None, strides=(2,2), padding='same' ))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output_conv = tf.keras.layers.LeakyReLU(alfa_relu)(x)

    x = tf.keras.layers.Flatten()(output_conv)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output_1=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(1, activation='sigmoid'), name='output_fake_real')(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=512, kernel_size=4, use_bias=False, activation=None, strides=(1,1), padding='same' ))(output_conv)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=1024, kernel_size=4, use_bias=False, activation=None, strides=(1,1), padding='same' ))(output_conv)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alfa_relu)(x)

    x = tf.keras.layers.Flatten()(output_conv)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    output_2 = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(num_classes, activation='softmax'), name='output_class_num')(x)

    model = tf.keras.Model(inputs=x_input, outputs=[output_1, output_2], name='discriminator')
    return model


def create_simple_adversarial_network(generator, discriminator, latent_space_input, labels_input):
    generator_output=generator([latent_space_input, labels_input])
    discriminator_outputs=discriminator(generator_output)
    model=tf.keras.Model(inputs=[latent_space_input, labels_input], outputs=discriminator_outputs)
    return model

if __name__ == "__main__":
    disc=create_simple_discriminator(x_input=tf.keras.Input(shape=(64,64,3)), num_classes=7)
    gen=create_simple_generator(input_x=tf.keras.Input(shape=(128,)), input_y=tf.keras.Input(shape=(7,)), discriminator_output_map_shape=(2,2,256), dropout_rate=0.2)
    create_simple_adversarial_network(gen, disc, tf.keras.Input(shape=(128,)), tf.keras.Input(shape=(7,)))