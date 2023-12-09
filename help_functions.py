import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit
from qibo.symbols import Z
from qibo import hamiltonians

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

    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

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
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    c.add(gates.CZ(0, 9))
    """
    for i in range(int(nqubits/2)):
        c.add(gates.CZ(i, nqubits-(i+1)))
    """

    return c


def single_image(vparams, image, label, layers, nqubits):
    # resizing and using rows of params and image
    row_image = tf.split(image, num_or_size_splits=10, axis=0)
    row_vparams = tf.split(vparams, num_or_size_splits=20, axis=0)

    # Creation: encoding block, variational block, measurement block
    ce = encoding_block(nqubits)
    cv = variational_block(nqubits)
    cm = measure_block(nqubits)

    # building the circuit
    # tensor_size = 2**nqubits
    # tensor_values = [1] + [0] * (tensor_size - 1)
    # initial_state = tf.constant(tensor_values, dtype=tf.float32)
    initial_state = 0
    for i in range(layers):
        print(f"Layer {i}")
        row_image_flat = tf.reshape(row_image[i], [-1])
        row_vparams_flat = tf.reshape(row_vparams[i], [-1])

        # encoding block
        if i == 0:
            ce.set_parameters(row_image_flat)
            result_ce = ce()

        else:
            ce.set_parameters(row_image_flat)
            result_ce = ce(initial_state)

        # variational block
        cv.set_parameters(row_vparams_flat)
        result_cv = cv(result_ce.state())
        state_cv = result_cv.state()
        initial_state = state_cv

    # stato finale dopo i layer di encoding e variational
    final_state = initial_state

    # measuring block
    print("Measurement")
    shots = 2
    result = cm(final_state, nshots=shots)

    # expectation values
    print("Calculating expectation values")
    expectation_values = []
    for k in range(nqubits):
        symbolic_ham = Z(k)
        ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
        expectation_value = ham.expectation_from_samples(result.frequencies())
        expectation_values.append(expectation_value)

    # softmax
    soft_output = tf.nn.softmax(expectation_values)
    print(f"Softmax: {soft_output}")

    return tf.keras.losses.CategoricalCrossentropy()(label, soft_output)


def loss_function(vparams, x_train, y_train, layers, nqubits):
    cf = 0
    counter = 0
    for x, y in zip(x_train, y_train):
        cf += single_image(vparams, x, y, layers, nqubits)
        print(f"Immagine {counter+1} processata")
        counter += 1
    cf /= len(x_train)
    return cf
