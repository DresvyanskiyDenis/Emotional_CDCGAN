import tensorflow as tf
import numpy as np
import pandas as pd


def create_mobilenet_model(input_dim=(224, 224, 3), nb_classes=8):
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


def build_descriminator(inputs, labels, image_size):
    pretrained_net_input=(image_size, image_size, 3)
