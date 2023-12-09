import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit
set_backend("tensorflow")

def label_converter(label):
    v = np.zeros(10)
    v[label] = 1
    return v

def initialize_data(train_size, resize, filt):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if train_size != 0:
        x_train = x_train[0:train_size]
        y_train = y_train[0:train_size]
        x_test = x_test[train_size + 1 : (train_size + 1) * 2]
        y_test = y_test[train_size + 1 : (train_size + 1) * 2]

    if filt == "yes":
        mask_train = (y_train == 0) | (y_train == 1)
        mask_test = (y_test == 0) | (y_test == 1)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]

    # Resize images
    width, length = 10, 10
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)


    x_train = tf.image.resize(x_train, [width, length])
    x_test = tf.image.resize(x_test, [width, length])

    # Normalize pixel values to be between 0 and pi
    x_train = x_train / 255.0 * np.pi
    x_test = x_test / 255.0 * np.pi

    #plt.imshow(x_train[0], cmap='gray')
    #plt.show()

    return x_train, y_train, x_test, y_test

def measure_block(nqubits):
    c = Circuit(nqubits)
    for i in range(nqubits):
        c.add(gates.M(i))
    return c

def encoding_block(nqubits):
    c = Circuit(nqubits)
    for i in range(nqubits):
        c.add(gates.RX(i, theta=0))
    return c

def variational_block(nqubits):
    c = Circuit(nqubits)
    for i in range(nqubits):
        c.add(gates.RY(i, theta=0))
        c.add(gates.RZ(i, theta=0))
    for i in range(nqubits-1):
        c.add(gates.CZ(i, i+1))

    c.add(gates.CZ(0, 9))
    '''
    for i in range(int(nqubits/2)):
        c.add(gates.CZ(i, nqubits-(i+1)))
    '''

    return c