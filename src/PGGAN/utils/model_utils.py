import tensorflow as tf


from src.PGGAN.utils.custom_layers import WeightedSum, MinibatchStdev, PixelNormalization



def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                tf.keras.backend.set_value(layer.alpha, alpha)


# add a discriminator block
def add_discriminator_block(old_model, n_input_layers=3):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = tf.keras.constraints.max_norm(1.0)
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
    in_image = tf.keras.layers.Input(shape=input_shape)
    # define new input processing layer
    d = tf.keras.layers.Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # define new block
    d = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.AveragePooling2D()(d)
    block_new = d
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model1 = tf.keras.Model(in_image, d)
    # compile model
    model1.compile(loss=wasserstein_loss,
                   optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
                  metrics=[binary_accuracy])
    # downsample the new larger image
    downsample = tf.keras.layers.AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model2 = tf.keras.Model(in_image, d)
    # compile model
    model2.compile(loss=wasserstein_loss,
                   optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
                  metrics=[binary_accuracy])
    return [model1, model2]


# define the discriminator models for each image resolution
def define_discriminator(n_blocks, input_shape=(4, 4, 3)):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = tf.keras.constraints.max_norm(1.0)
    model_list = list()
    # base model input
    in_image = tf.keras.layers.Input(shape=input_shape)
    # conv 1x1
    d = tf.keras.layers.Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = tf.keras.layers.Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = tf.keras.layers.Flatten()(d)
    out_class = tf.keras.layers.Dense(1)(d)
    # define model
    model = tf.keras.Model(in_image, out_class)
    # compile model
    model.compile(loss=wasserstein_loss,
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
                  metrics=[binary_accuracy])
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    return model_list


def add_generator_block(old_model):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = tf.keras.constraints.max_norm(1.0)
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = tf.keras.layers.UpSampling2D()(block_end)
    g = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(
        upsampling)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    g = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = tf.keras.layers.Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model1 = tf.keras.Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = tf.keras.Model(old_model.input, merged)
    return [model1, model2]


# define generator models
def define_generator(latent_dim, n_blocks, in_dim=4):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = tf.keras.constraints.max_norm(1.0)
    model_list = list()
    # base model latent input
    in_latent = tf.keras.layers.Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = tf.keras.layers.Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = tf.keras.layers.Reshape((in_dim, in_dim, 128))(g)
    # conv 4x4, input block
    g = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = tf.keras.layers.Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model = tf.keras.Model(in_latent, out_image)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    return model_list


# define composite models for training generators via discriminators
def define_composite(discriminators, generators):
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        d_models[0].trainable = False
        model1 = tf.keras.Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss,
                       optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
                  metrics=[binary_accuracy])
        # fade-in model
        d_models[1].trainable = False
        model2 = tf.keras.Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss,
                       optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
                  metrics=[binary_accuracy])
        # store
        model_list.append([model1, model2])
    return model_list
