import tensorflow as tf
import numpy as np
import pandas as pd


def create_mobilenet_model(input_dim=(224, 224, 3)):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_dim,
        alpha=1.0,
        include_top=False,
        weights="imagenet",
    )
    pooling = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    dense_1 = tf.keras.layers.Dense(1024, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00001))(pooling)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
    mobilenet_model = tf.keras.Model(inputs=model.inputs, outputs=output)
    del model
    return mobilenet_model


def build_descriminator_based_mobile_net(inputs, labels, image_size):
    pretrained_net_input=(image_size, image_size, 3)

def build_mnist_generator(inputs, labels, output_size_descriminator):
    x=tf.keras.layers.concatenate([inputs, labels], axis=1)
    x=tf.keras.layers.Dense(output_size_descriminator[0]*output_size_descriminator[1]*output_size_descriminator[2], activation='relu')(x)
    x=tf.keras.layers.Reshape((output_size_descriminator))(x)

    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=3, activation=None, strides=2, padding='same')(x)

    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=3, activation=None, strides=2, padding='same', output_padding=0)(x)

    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3, activation=None, strides=2, padding='same')(x)

    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3, activation=None, strides=2, padding='same')(x)

    # end
    x=tf.keras.layers.Activation('sigmoid')(x)

    generator_model=tf.keras.Model(inputs=[inputs, labels], outputs=x)
    return generator_model

def build_mnist_descriminator(inputs, labels, image_size):
    y=tf.keras.layers.Dense(image_size*image_size*1)(labels)
    y=tf.keras.layers.Reshape((image_size, image_size, 1))(y)
    concat_input=tf.keras.layers.concatenate([inputs,y], axis=-1)
    x=tf.keras.layers.LeakyReLU(0.2)(concat_input)
    x=tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation=None, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x=tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation=None, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x=tf.keras.layers.Conv2D(filters=128,kernel_size=4, activation=None, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x=tf.keras.layers.Conv2D(filters=256,kernel_size=4, activation=None, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # dense and sigmoid
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1, activation='sigmoid')(x)
    descriminator=tf.keras.Model(inputs=[inputs, labels], outputs=x)
    return descriminator

def build_mnist_adversarial_model(generator, descriminator, gen_input, descriminator_input, labels_input):
    adversarial_model=tf.keras.Model(inputs=[gen_input, labels_input], outputs=descriminator([generator([gen_input, labels_input]),labels_input]))
    adversarial_model.compile(optimizer='Adam', loss='binary_crossentropy')
    return adversarial_model

def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train=tf.keras.utils.to_categorical(y_train.reshape((-1,1)))
    y_test = tf.keras.utils.to_categorical(y_test.reshape((-1, 1)))
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

    # params
    latent_space_shape=100
    num_classes=y_train.shape[-1]
    image_size=x_train.shape[1]


    # generator_model
    input_gen=tf.keras.layers.Input((latent_space_shape,1))
    input_y=tf.keras.layers.Input((num_classes,1))
    gen_model=build_mnist_generator(inputs=input_gen,
                                    labels=input_y,
                                    output_size_descriminator=(2,2,256))

    # descriminator model
    input_desc=tf.keras.layers.Input((image_size, image_size,1))
    descriminator_model=build_mnist_descriminator(inputs=input_desc,
                                                  labels=input_y,
                                                  image_size=image_size)
    descriminator_model.compile(optimizer='Adam', loss='binary_crossentropy')

    # adversarial network
    descriminator_model.trainable=False
    adversarial_model=build_mnist_adversarial_model(gen_model, descriminator_model, input_gen, input_desc,input_y)
    adversarial_model.compile(optimizer='Adam', loss='binary_crossentropy')

    # summaries
    gen_model.summary()
    tf.keras.utils.plot_model(gen_model, show_shapes=True, to_file="model_gen.png")
    descriminator_model.summary()
    tf.keras.utils.plot_model(descriminator_model, show_shapes=True, to_file="model_disc.png")
    adversarial_model.summary()
    tf.keras.utils.plot_model(adversarial_model, show_shapes=True, to_file="model_advers.png")


    # train process




if __name__ == "__main__":
    # generator
    ''' x_input=tf.keras.layers.Input((100,1))
    y_input=tf.keras.layers.Input((10,1))
    gen_model=build_mnist_generator(x_input, y_input, 28)
    tf.keras.utils.plot_model(gen_model, show_shapes=True, to_file="model_gen.png")

    # discriminator
    x_input=tf.keras.layers.Input((28,28,1))
    descrim_model=build_mnist_descriminator(x_input, y_input, 28)
    tf.keras.utils.plot_model(descrim_model, show_shapes=True, to_file="model_disc.png")

    # Adversarial model
    adversarial_model=build_mnist_adversarial_model(gen_model, descrim_model, y_input)
    tf.keras.utils.plot_model(adversarial_model, show_shapes=True, to_file="model_advers.png")'''
    train()