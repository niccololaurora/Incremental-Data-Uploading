import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, Circuit
from qibo.optimizers import optimize
from qibo.symbols import Z
from qibo import hamiltonians
from help_functions import initialize_data, measure_block, variational_block, encoding_block, label_converter
set_backend("tensorflow")


def loss_function(vparams, x_train, y_train, layers, nqubits):
    cf = 0
    counter = 0
    for x, y in zip(x_train, y_train):
        cf += single_image(vparams, x, y, layers, nqubits)
        print(f"Immagine {counter+1} processata")
        counter += 1
    cf /= len(x_train)
    return cf 


def single_image(vparams, image, label, layers, nqubits):    
    # resizing and using rows of params and image
    row_image = tf.split(image, num_or_size_splits=10, axis=0)
    row_vparams = tf.split(vparams, num_or_size_splits=20, axis=0)

    # building the circuit
    tensor_size = 2**nqubits
    tensor_values = [1] + [0] * (tensor_size - 1)
    initial_state = tf.constant(tensor_values, dtype=tf.float32)
    for i in range(layers):
        print(f"Layer {i}")
        row_image_flat = tf.reshape(row_image[i], [-1])
        row_vparams_flat = tf.reshape(row_vparams[i], [-1])

        # encoding block
        ce = encoding_block(nqubits)
        ce.set_parameters(row_image_flat)
        result_ce = ce(initial_state)

        # variational block
        cv = variational_block(nqubits)
        cv.set_parameters(row_vparams_flat)
        result_cv = cv(result_ce.state())
        state_cv = result_cv.state()
        

    # stato finale dopo i layer di encoding e variational
    final_state = state_cv

    # measuring block
    print("Measurement")
    shots = 2
    c = measure_block(nqubits)
    result = c(final_state, nshots=shots)

    # expectation values
    print("Calculating expectation values")
    expectation_values = []
    for k in range(nqubits):
        ham = hamiltonians.Hamiltonian(k, matrix=gates.Z)
        expectation_value = ham.expectation_from_samples(result.frequencies())
        expectation_values.append(expectation_value)

    # softmax
    soft_output = tf.nn.softmax(expectation_values)
    print(f"Softmax: {soft_output}")
    
    return tf.keras.losses.CategoricalCrossentropy()(label, soft_output)


# ==================================
# Inizializzazione
# ==================================
nome_file = 'mnist_reuploading.txt'
nepochs = 3
nqubits = 10
layers = 10
nclasses = 2
train_size = 2
batch_size = 2
resize = 10
filt = "no"
loss = loss_function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vparams = tf.random.normal(shape=(400,))


# Data loading and filtering
x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)

print(f"{x_train.shape}")
y_train = np.array([label_converter(label) for label in y_train])
y_test = np.array([label_converter(label) for label in y_test])


# ==================================
# STAMPA INFO SU FILE
# ==================================
with open(nome_file, 'a') as file:
    print(f"Parametri iniziali circuito", file=file)
    print(f"{vparams}", file=file)
    print("="*60, file=file)


# perform optimization
for i in range(nepochs):
    print(f"Epoch {i+1}")
    with tf.GradientTape() as tape:
        tape.watch(vparams)
        l = loss(vparams, x_train, y_train, layers, nqubits)
    grads = tape.gradient(l, vparams)

    with open(nome_file, 'a') as file:
        print(f"Epoch {i+1}", file=file)
        print(f"Gradienti\n {grads}", file=file)
        print("="*60, file=file)

    optimizer.apply_gradients(zip([grads], [vparams]))



print("Fine del training")
with open(nome_file, 'a') as file:
    print("/"*60, file=file)
    print("/"*60, file=file)
    print("Fine del training", file=file)
    print("/"*60, file=file)
    print("/"*60, file=file)
    print(f"Parametri finali circuito", file=file)
    print(f"{params}", file=file)
    print(f"Loss {best}", file=file)
    print("="*60, file=file)
