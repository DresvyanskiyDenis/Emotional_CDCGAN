from src.cgan import build_and_train_models
import tensorflow as tf

from src.models import create_mobilenet_model

if __name__ == "__main__":
    #build_and_train_models()
    #model=create_mobilenet_model(input_dim=(224, 224, 3))
    #tf.keras.utils.plot_model(model, show_shapes=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)
