import tensorflow as tf
import os
from src.EmoCDCGAN.models import create_simple_generator, create_simple_discriminator


class ACGAN():
    generator:tf.keras.Model
    discriminator:tf.keras.Model
    adversarial:tf.keras.Model



    def __init__(self):
        pass

    def create_generator(self, z_input_shape:tuple, c_input_shape:tuple, dropout_rate:float=0.2):
        input_x_gen = tf.keras.layers.Input(z_input_shape)
        input_y = tf.keras.layers.Input(c_input_shape)
        generator_model = create_simple_generator(input_x_gen, input_y, dropout_rate=dropout_rate)
        self.generator=generator_model
        return generator_model

    def create_discriminator(self, x_input_shape:tuple, num_classes:int, dropout_rate=0.2):
        input_x_disc = tf.keras.layers.Input(x_input_shape)
        discriminator_model = create_simple_discriminator(x_input=input_x_disc, num_classes=num_classes,
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