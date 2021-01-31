import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


def train_n_mini_batches(model, data, labels, batch_size=4,
                         loss=tf.keras.losses.categorical_crossentropy):
    tf.executing_eagerly()
    # print((model.name))
    if model.name == 'discriminator':
        model.trainable = True
    train_vars = model.trainable_variables
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    total_loss = 0
    num_mini_batches=data[0].shape[0] // batch_size
    for i in range(num_mini_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        train_data = [data[0][start_idx:end_idx], data[1][start_idx:end_idx]]
        train_labels = labels[start_idx:end_idx].reshape((-1, 1)).astype('float32')
        with tf.GradientTape() as tape:
            predictions = model(train_data, training=True)
            loss_value = loss(train_labels, predictions)
        total_loss += loss_value.numpy().mean()

        gradients = tape.gradient(loss_value, train_vars)
        accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradient, gradients)]

    accum_gradient = [this_grad / num_mini_batches for this_grad in accum_gradient]

    model.optimizer.apply_gradients(zip(accum_gradient, train_vars))

    batch_loss = total_loss / num_mini_batches
    if model.name == 'discriminator':
        model.trainable = False
    return batch_loss


def shuffle(x, y):
    permut = np.random.permutation(x.shape[0])
    return x[permut], y[permut]


def train_(model, data, labels, batch_size=4,
                         loss=tf.keras.losses.categorical_crossentropy):
    tf.executing_eagerly()
    train_vars = model.trainable_variables
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    total_loss = 0
    num_mini_batches=data.shape[0] // batch_size
    for i in range(num_mini_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        train_data = data[start_idx:end_idx]
        train_labels = labels[start_idx:end_idx].astype('float32')
        with tf.GradientTape() as tape:
            predictions = model(train_data, training=True)
            loss_value = loss(train_labels, predictions)
        total_loss += loss_value.numpy().mean()

        gradients = tape.gradient(loss_value, train_vars)
        accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradient, gradients)]

    accum_gradient = [this_grad / num_mini_batches for this_grad in accum_gradient]

    model.optimizer.apply_gradients(zip(accum_gradient, train_vars))

    batch_loss = total_loss / num_mini_batches
    return batch_loss


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype('float32')
    x_train = x_train / 255.
    y_train = tf.keras.utils.to_categorical(y_train)

    x_val = x_train[:250]
    y_val = y_train[:250]
    x_train = x_train[250:15000]
    y_train = y_train[250:15000]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 4, strides=(1, 1), activation='relu', padding='same', input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, 3, strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy')
    model.summary()

    # val check
    predictions = model.predict(x_val)
    print('accuracy_score before train:%f' % accuracy_score(np.argmax(y_val, axis=-1), np.argmax(predictions, axis=-1)))

    for epoch in range(1000):
        x_train, y_train = shuffle(x_train, y_train)
        batch_size=256
        mini_batch_size=64
        total_loss=0
        for i in range(x_train.shape[0]//batch_size):

            batch_loss = train_(model=model, data=x_train[i*batch_size:(i+1)*batch_size], labels=y_train[i*batch_size:(i+1)*batch_size],
                                          batch_size=mini_batch_size,
                                          loss=tf.keras.losses.categorical_crossentropy)
            total_loss += batch_loss
        # print('batch loss:%f'%batch_loss)
        total_loss /= (x_train.shape[0]//batch_size)
        if epoch % 1 == 0:
            predictions = model.predict(x_val)
            print('accuracy_score epoch %i train acc:%f, loss:%f' % (epoch, accuracy_score(np.argmax(y_val, axis=-1),
                                                                              np.argmax(predictions, axis=-1)), total_loss))


