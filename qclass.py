import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit, hamiltonians
from qibo.symbols import Z, I
from qibo.optimizers import optimize, sgd, cmaes
from help_functions import batch_data, calculate_batches, label_converter

set_backend("tensorflow")


class MyClass:
    def __init__(
        self,
        train_size,
        epochs,
        batch_size,
        method,
        learning_rate,
        nome_file,
        layers,
    ):
        self.nome_file = nome_file
        self.epochs = epochs
        self.layers = layers
        self.train_size = train_size
        self.batch_size = batch_size
        self.method = method
        self.learning_rate = learning_rate
        self.nqubits = 10
        self.epochs_early_stopping = epochs
        self.splitting = 10
        self.nclasses = 2
        self.test_size = train_size
        self.validation_split = 0.2
        self.tolerance = 1e-4
        self.patience = 10
        self.resize = 10
        self.filt = "no"
        self.vparams = np.random.normal(loc=0, scale=1, size=(20 * self.layers, 20))
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.batch_x = 0
        self.batch_y = 0
        self.x_validation = 0
        self.y_validation = 0
        self.index = 0
        self.options = {
            "optimizer": self.method,
            "learning_rate": self.learning_rate,
            "nepochs": 1,
            "nmessage": 5,
        }

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
            validation_size = int(len(x_train) * self.validation_split)

            x_validation = x_train[:validation_size]
            y_validation = y_train[:validation_size]
            x_train = x_train[validation_size:]
            y_train = y_train[validation_size:]
            x_test = x_test[0 : self.test_size]
            y_test = y_test[0 : self.test_size]

        # Resize images
        width, length = self.resize, self.resize

        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)
        x_validation = tf.expand_dims(x_validation, axis=-1)

        x_train = tf.image.resize(x_train, [width, length])
        x_test = tf.image.resize(x_test, [width, length])
        x_validation = tf.image.resize(x_validation, [width, length])

        # Normalize pixel values to be between 0 and 1
        x_train = x_train / 255.0 * np.pi
        x_test = x_test / 255.0 * np.pi
        x_validation = x_validation / 255.0 * np.pi

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_validation = x_validation
        self.y_validation = y_validation

        # Batching
        number_of_batches, sizes_batches = calculate_batches(
            self.x_train, self.batch_size
        )
        self.batch_x, self.batch_y = batch_data(
            self.x_train,
            self.y_train,
            number_of_batches,
            sizes_batches,
        )

    def measure_block(self):
        c = Circuit(self.nqubits)
        for i in range(self.nqubits):
            c.add(gates.M(i))
        return c

    def entanglement_block(self):
        """
        Args: None
        Return: circuit with CZs
        """
        c = Circuit(10)
        for q in range(0, 9, 2):
            c.add(gates.CNOT(q, q + 1))
        for q in range(1, 8, 2):
            c.add(gates.CNOT(q, q + 1))
        c.add(gates.CNOT(9, 0))
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

    def test_loop(self):
        predictions = []
        for x in self.x_test:
            output = self.circuit(x)
            predictions.append(output)

        accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        accuracy.update_state(self.y_test, predictions)
        return accuracy

    def validation_loop(self):
        predictions = []
        for x in self.x_validation:
            output = self.circuit(x)
            predictions.append(output)

        loss_value = tf.keras.losses.CategoricalCrossentropy()(
            self.y_validation, predictions
        )
        return loss_value

    def training_loop(self):
        if (
            (self.method == "Adadelta")
            or (self.method == "Adagrad")
            or (self.method == "Adam")
        ):
            best, params, extra = 0, 0, 0
            epoch_loss = []
            for i in range(self.epochs):
                with open(self.nome_file, "a") as file:
                    print("=" * 60, file=file)
                    print(f"Epoch {i+1}", file=file)

                batch_loss = []
                for k in range(len(self.batch_x)):
                    best, params, extra = optimize(
                        self.loss_function,
                        self.vparams,
                        args=(self.batch_x[k], self.batch_y[k]),
                        method="sgd",
                        options=self.options,
                    )
                    batch_loss.append(best)

                    with open(self.nome_file, "a") as file:
                        print("/" * 60, file=file)
                        print(f"Batch {k+1}", file=file)
                        print(f"Parametri:\n{params[0:20]}", file=file)
                        print("/" * 60, file=file)

                e_loss = sum(batch_loss) / len(batch_loss)
                epoch_loss.append(e_loss)

                # Validation
                validation_loss = self.validation_loop()
                epoch_validation_loss.append(validation_loss)

            with open(self.nome_file, "a") as file:
                print("=" * 60, file=file)
                print(f"Parametri finali:\n{params[0:20]}", file=file)
                print("=" * 60, file=file)

        else:
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method="parallel_L-BFGS-B",
            )

        return epoch_loss, epoch_validation_loss, params, self.epochs_early_stopping

    def early_stopping(self, training_loss_history, validation_loss_history):
        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(len(training_loss_history)):
            training_loss = training_loss_history[epoch]
            validation_loss = validation_loss_history[epoch]

            # Verifica se la loss di validazione ha migliorato
            if validation_loss < best_validation_loss - self.tolerance:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                with open(self.nome_file, "a") as file:
                    print(">" * 60, file=file)
                    print(f"Early stopping at epoch {epoch + 1}.")
                    print(">" * 60, file=file)
                self.epochs_early_stopping = epoch + 1
                return True

        return False

    def loss_function(self, vparams, batch_x, batch_y):
        if vparams is None:
            vparams = self.vparams
        self.set_parameters(vparams)

        predictions = []
        for x in batch_x:
            output = self.circuit(x)
            predictions.append(output)

        loss_value = tf.keras.losses.CategoricalCrossentropy()(batch_y, predictions)
        return loss_value

    def circuit(self, image):
        # Rows from image
        rows = self.rows_creator(image)

        # Creation: encoding block, variational block, measurement block, entanglement block
        ce = self.encoding_block()
        cv = self.variational_block()
        cm = self.measure_block()
        cent = self.entanglement_block()

        # Initial state
        tensor_size = 2**self.nqubits
        tensor_values = [1] + [0] * (tensor_size - 1)
        initial_state = tf.constant(tensor_values, dtype=tf.float32)

        for j in range(self.layers):
            for i in range(self.splitting):
                # encoding block
                ce.set_parameters(rows[i])
                result_ce = ce(initial_state)

                # variational block
                cv.set_parameters(self.vparams[i + 20 * j])
                result_cv = cv(result_ce.state())

                # entanglement block
                result_cent = cent(result_cv.state())

                # State
                initial_state = result_cent.state()

        # stato finale dopo i layer di encoding e variational
        final_state = initial_state

        # measuring block
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
