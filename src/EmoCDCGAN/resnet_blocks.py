import tensorflow as tf

def conv_block(input, num_filters, stride):
    f1, f2 = num_filters

    # first conv, 1x1, stride reduction
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, activation=None, stride=stride, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=3, activation=None, stride=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, activation=None, stride=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut connection, but with 1x1 conv to decrease the feature map
    shortcut = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, activation=None, stride=stride, padding='same')(input)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # connection
    add=tf.keras.layers.Add([shortcut, x])
    add=tf.keras.layers.ReLU()(add)

    return add

def resnet_identity(input, num_filters):
    # TODO: make it done
    pass
