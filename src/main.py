from src.cgan import build_and_train_models
import tensorflow as tf

def expand_label_input(x):
    x = tf.keras.backend.expand_dims(x, axis = 1)
    x = tf.keras.backend.expand_dims(x, axis = 1)
    x = tf.keras.backend.tile(x, [1, 32, 32, 1])
    return x

def build_discriminator():
    """
    Discriminator Network
    """

    input_shape = (64, 64, 3)
    label_shape = (6,)
    image_input = tf.keras.layers.Input(shape=input_shape)
    label_input = tf.keras.layers.Input(shape=label_shape)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    label_input1 = tf.keras.layers.Lambda(expand_label_input)(label_input)
    x = tf.keras.layers.concatenate([x, label_input1], axis=3)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[image_input, label_input], outputs=[x])
    return model


if __name__ == "__main__":
    #build_and_train_models()
    model=build_discriminator()
    tf.keras.utils.plot_model(model, show_shapes=True)