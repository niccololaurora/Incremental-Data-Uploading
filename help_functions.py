import tensorflow as tf
import numpy as np


def label_converter(label):
    v = np.zeros(10)
    v[label] = 1
    return v


def initialize_data(train_size, resize, filt):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if filt == "yes":
        mask_train = (y_train == 0) | (y_train == 1)
        mask_test = (y_test == 0) | (y_test == 1)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]

    if train_size != 0:
        x_train = x_train[0:train_size]
        y_train = y_train[0:train_size]
        x_test = x_test[train_size + 1 : (train_size + 1) * 2]
        y_test = y_test[train_size + 1 : (train_size + 1) * 2]

    # Resize images
    width, length = 10, 10

    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    x_train = tf.image.resize(x_train, [width, length])
    x_test = tf.image.resize(x_test, [width, length])

    # Normalize pixel values to be between 0 and pi
    x_train = x_train / 255.0 * np.pi
    x_test = x_test / 255.0 * np.pi

    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()
    # Fixing the
    y_train = np.array([label_converter(label) for label in y_train])
    y_test = np.array([label_converter(label) for label in y_test])

    return x_train, y_train, x_test, y_test
