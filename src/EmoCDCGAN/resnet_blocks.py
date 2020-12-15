import tensorflow as tf

def resnet_conv_block(input, num_filters, stride):
    f1, f2 = num_filters

    # first conv, 1x1, stride reduction
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=3, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut connection, but with 1x1 conv to decrease the feature map
    shortcut = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # connection
    add=tf.keras.layers.Add()([shortcut, x])
    add=tf.keras.layers.ReLU()(add)

    return add

def resnet_identity(input, num_filters):
    f1, f2=num_filters

    # first conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=3, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut connection
    add=tf.keras.layers.Add()([input, x])
    add=tf.keras.layers.ReLU()(add)

    return add

def resnet_conv_transpose(input, num_filters, stride):
    f1,f2 = num_filters

    # first conv, 1x1, stride reduction
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=1, activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=3, activation=None, strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, activation=None, strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut connection, but with 1x1 conv to decrease the feature map
    shortcut = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # connection
    add = tf.keras.layers.Add()([shortcut, x])
    add = tf.keras.layers.ReLU()(add)

    return add

def resnet_identity_transponse(input, num_filters):
    f1, f2=num_filters

    # first conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=1, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=3, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut connection
    add=tf.keras.layers.Add()([input, x])
    add=tf.keras.layers.ReLU()(add)

    return add

def create_discriminator_resnet_based(x_input, y_input, image_shape):
    # 224x224x3
    dense_y=tf.keras.layers.Dense(image_shape*image_shape*1, activation=None)(y_input)
    reshape=tf.keras.layers.Reshape((image_shape,image_shape,1))(dense_y)
    concat=tf.keras.layers.concatenate([x_input, reshape])
    concat=tf.keras.layers.ReLU()(concat)

    x = resnet_conv_block(input=concat, num_filters=(64, 256), stride=(2, 2))
    x = resnet_identity(input=x, num_filters=(64, 256))
    x = resnet_identity(input=x, num_filters=(64, 256))

    x = resnet_conv_block(input=x, num_filters=(128, 512), stride=(2, 2))
    x = resnet_identity(input=x, num_filters=(128, 512))
    x = resnet_identity(input=x, num_filters=(128, 512))

    x = resnet_conv_block(input=x, num_filters=(256, 1024), stride=(2, 2))
    x = resnet_identity(input=x, num_filters=(256, 1024))
    x = resnet_identity(input=x, num_filters=(256, 1024))

    x = resnet_conv_block(input=x, num_filters=(512, 1024), stride=(2, 2))
    x = resnet_identity(input=x, num_filters=(512, 1024))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation=None, strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x=tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model=tf.keras.Model(inputs=[x_input, y_input], outputs=x)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='discriminator_resnet_based_model.png')
    return model

def create_generator_resnet_based(x_input, y_input, discriminator_output_map_shape=(7,7,64)):
    concat=tf.keras.layers.concatenate([x_input, y_input])
    x=tf.keras.layers.Flatten()(concat)
    x=tf.keras.layers.Dense(discriminator_output_map_shape[0]*
                            discriminator_output_map_shape[1]*
                            discriminator_output_map_shape[2], activation=None)(x)
    x=tf.keras.layers.Reshape(discriminator_output_map_shape)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    x = resnet_conv_transpose(input=x, num_filters=(64, 256), stride=(2, 2))
    x = resnet_identity_transponse(input=x, num_filters=(64, 256))
    x = resnet_identity_transponse(input=x, num_filters=(64, 256))

    x = resnet_conv_transpose(input=x, num_filters=(128, 512), stride=(2, 2))
    x = resnet_identity_transponse(input=x, num_filters=(128, 512))
    x = resnet_identity_transponse(input=x, num_filters=(128, 512))

    x = resnet_conv_transpose(input=x, num_filters=(256, 1024), stride=(2, 2))
    x = resnet_identity_transponse(input=x, num_filters=(256, 1024))
    x = resnet_identity_transponse(input=x, num_filters=(256, 1024))

    x = resnet_conv_transpose(input=x, num_filters=(512, 1024), stride=(2, 2))
    x = resnet_identity_transponse(input=x, num_filters=(512, 1024))

    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=7, activation=None, strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=7, activation=None, strides=(1,1), padding='same')(x)
    x = tf.keras.layers.Activation(tf.keras.activations.tanh)(x)

    model=tf.keras.Model(inputs=[input_x, input_y], outputs=x)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='generator_resnet_based_model.png')
    return model



if __name__ == "__main__":
    input_x=tf.keras.layers.Input((224,224,3))
    input_y=tf.keras.layers.Input((7))
    model=create_discriminator_resnet_based(input_x, input_y, 224)
    model.summary()

    input_x=tf.keras.layers.Input((100,))
    input_y = tf.keras.layers.Input((7,))
    model=create_generator_resnet_based(input_x, input_y)
    model.summary()