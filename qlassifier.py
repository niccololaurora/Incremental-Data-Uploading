import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit, hamiltonians
from qibo.symbols import Z
from qibo.optimizers import optimize, sgd, cmaes
from help_functions import (
    single_image,
    initialize_data,
    label_converter,
)

set_backend("tensorflow")


class MyClass:
    def __init__(train_size, resize, filt, epochs, batch_size, method):
        self.nqubits = 10
        self.epochs = 3
        self.layers = 10
        self.nclasses = 2
        self.train_size = train_size
        self.batch_size = batch_size
        self.resize = resize
        self.filt = filt
        self.method = method
        self.method = "sgd"
        self.optimizer = "Adam"
        self.learning_rate = 0.001
        self.vparams = tf.Variable(tf.random.normal(shape=(400,)), trainable=True)

    def measure_block(self):
        c = Circuit(self.nqubits)
        for i in range(nqubits):
            c.add(gates.M(i))
        return c

    def encoding_block(self):
        c = Circuit(self.nqubits)
        for i in range(self.nqubits):
            c.add(gates.RX(i, theta=0))
        return c

    def variational_block(self):
        c = Circuit(self.nqubits)
        for i in range(nqubits):
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

    def training_loop(self):
        if self.method == "sgd":
            # perform optimization
            options = {
                "optimizer": self.optimizer,
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

    def loss_function(self):
        predictions = []
        for x in self.x_train:
            output = single_image(x)
            predictions.append(output)

        loss_value = tf.keras.losses.CategoricalCrossentropy()(
            self.y_train, predictions
        )

        return loss_value

    def single_image(self, image):
        # resizing and using rows of params and image
        row_image = tf.split(image, num_or_size_splits=10, axis=0)
        row_vparams = tf.split(self.vparams, num_or_size_splits=20, axis=0)

        # Creation: encoding block, variational block, measurement block
        ce = self.encoding_block(self.nqubits)
        cv = self.variational_block(self.nqubits)
        cm = self.measure_block(self.nqubits)

        # building the circuit
        tensor_size = 2**self.nqubits
        tensor_values = [1] + [0] * (tensor_size - 1)
        initial_state = tf.constant(tensor_values, dtype=tf.float32)

        # initial_state = 0
        for i in range(self.layers):
            row_image_flat = tf.reshape(row_image[i], [-1])
            row_vparams_flat = tf.reshape(row_vparams[i], [-1])

            # encoding block
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
        shots = 2
        result = cm(final_state, nshots=shots)

        # expectation values
        expectation_values = []
        for k in range(nqubits):
            symbolic_ham = Z(k)
            ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
            expectation_value = ham.expectation_from_samples(result.frequencies())
            expectation_values.append(expectation_value)

        # softmax
        soft_output = tf.nn.softmax(expectation_values)

        return soft_output
