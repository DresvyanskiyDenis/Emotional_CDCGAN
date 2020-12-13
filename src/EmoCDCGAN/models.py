import tensorflow as tf
import numpy as np


def create_mobilenet_model(input_dim=(224, 224, 3)):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_dim,
        alpha=1.0,
        include_top=False,
        weights="imagenet",
    )
    tf.keras.utils.plot_model(model,show_shapes=True)
    pooling = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    dense_1 = tf.keras.layers.Dense(1024, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00001))(pooling)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
    mobilenet_model = tf.keras.Model(inputs=model.inputs, outputs=output)
    del model
    return mobilenet_model


def build_descriminator_based_mobile_net(inputs, labels, image_size, num_channels=3):
    pretrained_net_input=(image_size, image_size, num_channels)

def build_own_discriminator(inputs, labels, image_size, num_channels=3):
    


if __name__ == "__main__":
    create_mobilenet_model()