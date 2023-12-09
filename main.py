import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit
from qibo.optimizers import optimize
from qibo import hamiltonians
from help_functions import (
    initialize_data,
    measure_block,
    variational_block,
    encoding_block,
    label_converter,
    loss_function,
    single_image,
)

set_backend("tensorflow")


def main():
    # ==================================
    # Inizializzazione
    # ==================================
    nome_file = "mnist_reuploading.txt"
    nepochs = 3
    nqubits = 10
    layers = 10
    nclasses = 2
    train_size = 10
    batch_size = 2
    resize = 10
    filt = "no"
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    vparams = tf.random.normal(shape=(400,))

    # Data loading and filtering
    x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)
    with open(nome_file, "a") as file:
        print(f"Tipo: {type(y_train)}")
        print("=" * 60, file=file)

    y_train = np.array([label_converter(label) for label in y_train])
    y_test = np.array([label_converter(label) for label in y_test])

    # ==================================
    # STAMPA INFO SU FILE
    # ==================================
    with open(nome_file, "a") as file:
        print(f"Parametri iniziali circuito", file=file)
        print(f"{vparams}", file=file)
        print("=" * 60, file=file)

    # perform optimization
    for i in range(nepochs):
        print(f"Epoch {i+1}")
        with tf.GradientTape() as tape:
            tape.watch(vparams)
            l = loss_function(vparams, x_train, y_train, layers, nqubits)
        grads = tape.gradient(l, vparams)

        with open(nome_file, "a") as file:
            print(f"Epoch {i+1}", file=file)
            print(f"Gradienti\n {grads}", file=file)
            print("=" * 60, file=file)

        optimizer.apply_gradients(zip([grads], [vparams]))

    print("Fine del training")
    with open(nome_file, "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print("Fine del training", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Parametri finali circuito", file=file)
        print(f"{params}", file=file)
        print(f"Loss {best}", file=file)
        print("=" * 60, file=file)


if __name__ == "__main__":
    main()
