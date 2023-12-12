import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit, hamiltonians
from qibo.symbols import Z, I
from qibo.optimizers import optimize, sgd, cmaes
from help_functions import (
    label_converter,
)

set_backend("tensorflow")


class MyClass:
    def __init__(self, train_size, resize, filt, epochs, batch_size, method):
        self.nqubits = 10
        self.epochs = 3
        self.layers = 10
        self.nclasses = 2
        self.train_size = train_size
        self.batch_size = batch_size
        self.resize = resize
        self.filt = filt
        self.method = method
        self.learning_rate = 0.001
        self.vparams = np.random.normal(loc=0, scale=1, size=(20, 20))
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.index = 0

    def create_hamiltonian(self, index):
        ham = (
            Z(index)
            * I(0)
            * I(1)
            * I(2)
            * I(3)
            * I(4)
            * I(5)
            * I(6)
            * I(7)
            * I(8)
            * I(9)
        )
        return hamiltonians.SymbolicHamiltonian(ham)

    def get_parameters(self):
        return self.vparams

    def set_parameters(self, vparams):
        self.vparams = vparams

    def initialize_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        if self.filt == "yes":
            mask_train = (y_train == 0) | (y_train == 1)
            mask_test = (y_test == 0) | (y_test == 1)
            x_train = x_train[mask_train]
            y_train = y_train[mask_train]
            x_test = x_test[mask_test]
            y_test = y_test[mask_test]

        if self.train_size != 0:
            x_train = x_train[0 : self.train_size]
            y_train = y_train[0 : self.train_size]
            x_test = x_test[self.train_size + 1 : (self.train_size + 1) * 2]
            y_test = y_test[self.train_size + 1 : (self.train_size + 1) * 2]

        # Resize images
        width, length = 10, 10

        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)

        x_train = tf.image.resize(x_train, [width, length])
        x_test = tf.image.resize(x_test, [width, length])

        # Normalize pixel values to be between 0 and pi
        x_train = x_train / 255.0 * np.pi
        x_test = x_test / 255.0 * np.pi

        y_train = np.array([label_converter(label) for label in y_train])
        y_test = np.array([label_converter(label) for label in y_test])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def measure_block(self):
        c = Circuit(self.nqubits)
        for i in range(self.nqubits):
            c.add(gates.M(i))
        return c

    def encoding_block(self):
        c = Circuit(self.nqubits)
        for i in range(self.nqubits):
            c.add(gates.RX(i, theta=0))
        return c

    def variational_block(self):
        c = Circuit(self.nqubits)
        for i in range(self.nqubits):
            c.add(gates.RY(i, theta=0))
            c.add(gates.RZ(i, theta=0))
        for i in range(self.nqubits - 1):
            c.add(gates.CZ(i, i + 1))

        c.add(gates.CZ(9, 0))
        """
        for i in range(int(nqubits/2)):
            c.add(gates.CZ(i, nqubits-(i+1)))
        """
        return c

    def rows_creator(self, image):
        rows = []
        for i in range(self.nqubits):
            row = image[i, :10]
            rows.append(row)

        return rows

    def training_loop(self):
        if (
            (self.method == "Adam")
            or (self.method == "Adagrad")
            or (self.method == "Adadelta")
        ):
            # perform optimization
            options = {
                "optimizer": self.method,
                "learning_rate": self.learning_rate,
                "nepochs": self.epochs,
                "nmessage": 1,
            }
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method=self.method,
                options=options,
            )

        else:
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method=self.method,
            )

        return best, params, extra

    def loss_function(self, vparams=None):
        if vparams is None:
            vparams = self.vparams
        self.set_parameters(vparams)

        predictions = []
        for x in self.x_train:
            output = self.single_image(x)
            predictions.append(output)

        loss_value = tf.keras.losses.CategoricalCrossentropy()(
            self.y_train, predictions
        )

        return loss_value

    def single_image(self, image):
        # resizing and using rows of params and image
        # row_image = tf.split(image, num_or_size_splits=10, axis=0)
        # row_vparams = tf.split(self.vparams, num_or_size_splits=20, axis=0)

        rows = self.rows_creator(image)

        # Creation: encoding block, variational block, measurement block
        ce = self.encoding_block()
        cv = self.variational_block()
        cm = self.measure_block()

        # building the circuit
        tensor_size = 2**self.nqubits
        tensor_values = [1] + [0] * (tensor_size - 1)
        initial_state = tf.constant(tensor_values, dtype=tf.float32)

        # initial_state = 0
        for i in range(self.layers):
            # row_image_flat = tf.reshape(rows[i], [-1])
            # row_vparams_flat = tf.reshape(row_vparams[i], [-1])

            # encoding block
            ce.set_parameters(rows[i])
            result_ce = ce(initial_state)

            # variational block
            cv.set_parameters(self.vparams[i])
            result_cv = cv(result_ce.state())
            state_cv = result_cv.state()
            initial_state = state_cv

        # stato finale dopo i layer di encoding e variational
        final_state = initial_state

        # measuring block
        shots = 2
        result = cm(final_state)

        # expectation values
        expectation_values = []

        for k in range(self.nqubits):
            hamiltonian = self.create_hamiltonian(k)
            expectation_value = hamiltonian.expectation(result.state())
            expectation_values.append(expectation_value)

        # softmax
        soft_output = tf.nn.softmax(expectation_values)

        return soft_output
