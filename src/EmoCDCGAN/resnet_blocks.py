import tensorflow as tf
import tensorflow_addons as tfa

def resnet_conv_block(input, num_filters, stride, group_norm_groups=32):
    f1, f2 = num_filters
    leaky_relu_par=0.2
    # first conv, 1x1, stride reduction
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, use_bias=False,activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=3, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)

    # shortcut connection, but with 1x1 conv to decrease the feature map
    shortcut = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    shortcut = tfa.layers.GroupNormalization(groups=group_norm_groups)(shortcut)

    # connection
    add=tf.keras.layers.Add()([shortcut, x])
    add=tf.keras.layers.LeakyReLU(leaky_relu_par)(add)
    return add

def resnet_identity(input, num_filters, group_norm_groups=32):
    f1, f2=num_filters
    leaky_relu_par = 0.2
    # first conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=3, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)

    # shortcut connection
    add=tf.keras.layers.Add()([input, x])
    add=tf.keras.layers.LeakyReLU(leaky_relu_par)(add)

    return add

def resnet_conv_transpose(input, num_filters, stride, group_norm_groups=32):
    f1,f2 = num_filters
    leaky_relu_par=0.2

    # first conv, 1x1, stride reduction
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=1, use_bias=False,activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=3, use_bias=False,activation=None, strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)

    # shortcut connection, but with 1x1 conv to decrease the feature map
    shortcut = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    shortcut = tfa.layers.GroupNormalization(groups=group_norm_groups)(shortcut)

    # connection
    add = tf.keras.layers.Add()([shortcut, x])
    add = tf.keras.layers.LeakyReLU(leaky_relu_par)(add)

    return add

def resnet_identity_transponse(input, num_filters, group_norm_groups=32):
    f1, f2=num_filters
    leaky_relu_par = 0.2

    # first conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=1, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # second conv, 3x3
    x = tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=3, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(leaky_relu_par)(x)

    # third conv, 1x1
    x = tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=1, use_bias=False,activation=None, strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)

    # shortcut connection
    add=tf.keras.layers.Add()([input, x])
    add=tf.keras.layers.LeakyReLU(leaky_relu_par)(add)

    return add

def create_discriminator_resnet_based(x_input, y_input, image_shape, group_norm_groups=64, dropout_rate=0.2):
    # 224x224x3
    noise=tf.keras.layers.GaussianNoise(0.1)(x_input)
    dense_y=tf.keras.layers.Dense(image_shape*image_shape*1, activation=None)(y_input)
    reshape=tf.keras.layers.Reshape((image_shape,image_shape,1))(dense_y)
    concat=tf.keras.layers.concatenate([noise, reshape])
    concat=tf.keras.layers.LeakyReLU(0.2)(concat)
    concat = tf.keras.layers.Dropout(dropout_rate)(concat)

    x = resnet_conv_block(input=concat, num_filters=(64, 128), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity(input=x, num_filters=(64, 128), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity(input=x, num_filters=(64, 256), group_norm_groups=group_norm_groups)

    x = resnet_conv_block(input=x, num_filters=(64, 256), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity(input=x, num_filters=(64, 256), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity(input=x, num_filters=(128, 512), group_norm_groups=group_norm_groups)

    x = resnet_conv_block(input=x, num_filters=(128, 512), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity(input=x, num_filters=(256, 1024), group_norm_groups=group_norm_groups)
    x = resnet_identity(input=x, num_filters=(128, 512), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = resnet_conv_block(input=x, num_filters=(256, 1024), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity(input=x, num_filters=(256, 1024), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation=None, strides=(2, 2), padding='same')(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x=tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model=tf.keras.Model(inputs=[x_input, y_input], outputs=x, name='discriminator')
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='discriminator_resnet_based_model.png')
    return model

def create_generator_resnet_based(input_x, input_y, discriminator_output_map_shape=(7,7,64), group_norm_groups=64, dropout_rate=0.2):
    noise = tf.keras.layers.GaussianNoise(0.1)(input_x)
    concat=tf.keras.layers.concatenate([noise, input_y])
    x=tf.keras.layers.Flatten()(concat)
    x=tf.keras.layers.Dense(discriminator_output_map_shape[0]*
                            discriminator_output_map_shape[1]*
                            discriminator_output_map_shape[2], activation=None)(x)
    x=tf.keras.layers.Reshape(discriminator_output_map_shape)(x)
    x=tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x=tf.keras.layers.LeakyReLU(0.2)(x)
    x=tf.keras.layers.Dropout(dropout_rate)(x)

    x = resnet_conv_transpose(input=x, num_filters=(64, 128), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity_transponse(input=x, num_filters=(64, 128), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity_transponse(input=x, num_filters=(64, 256), group_norm_groups=group_norm_groups)

    x = resnet_conv_transpose(input=x, num_filters=(64, 256), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity_transponse(input=x, num_filters=(64, 256), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity_transponse(input=x, num_filters=(128, 512), group_norm_groups=group_norm_groups)

    x = resnet_conv_transpose(input=x, num_filters=(128, 512), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity_transponse(input=x, num_filters=(128, 512), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    #x = resnet_identity_transponse(input=x, num_filters=(256, 1024), group_norm_groups=group_norm_groups)

    x = resnet_conv_transpose(input=x, num_filters=(256, 1024), stride=(2, 2), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = resnet_identity_transponse(input=x, num_filters=(256, 1024), group_norm_groups=group_norm_groups)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=7, activation=None, strides=(2, 2), padding='same')(x)
    x = tfa.layers.GroupNormalization(groups=group_norm_groups)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, activation=None, strides=(1,1), padding='same')(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    model=tf.keras.Model(inputs=[input_x, input_y], outputs=x)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='generator_resnet_based_model.png')
    return model

def build_adversarial_model_resnet_based(generator, discriminator, latent_space_input, labels_input):
    model=tf.keras.Model(inputs=[latent_space_input, labels_input], outputs=[discriminator([generator([latent_space_input, labels_input]), labels_input])], name='adversarial')
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