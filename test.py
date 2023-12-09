from qibo import gates, Circuit, set_backend
from qibo.symbols import X, Y, Z
import tensorflow as tf
import numpy as np
from qibo.symbols import X, Z
from qibo import hamiltonians

set_backend("tensorflow")

tensor_size = 2**10
tensor_values = [1] + [0] * (tensor_size - 1)
tensorflow_tensor = tf.constant(tensor_values, dtype=tf.float32)
print(tensorflow_tensor)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image = x_train[0]

row_image = tf.split(image, num_or_size_splits=28, axis=0)
print(f"{type(row_image)}")

print("Circuito 1")

c = Circuit(2)
c.add(gates.H(0))
c.add(gates.H(1))
c.add(gates.M(0))
c.add(gates.M(1))


symbolic_ham = sum(Z(i) for i in range(1))
ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)

result = c(nshots=1)
print("frequencies")
print(result.frequencies())
expectation_value = ham.expectation_from_samples(result.frequencies())
print("expectation_value")
print(expectation_value)


print("="*60)
print("Circuito 2")

c = Circuit(1)
c.add(gates.H(0))
c.add(gates.M(0))

symbolic_ham = Z(0)
ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)

result = c(nshots=1)
print("frequencies")
print(result.frequencies())
expectation_value = ham.expectation_from_samples(result.frequencies())
print("expectation_value")
print(expectation_value)


print("="*60)
print("Circuito 3")

c = Circuit(2)
c.add(gates.H(0))
c.add(gates.H(1))
c.add(gates.M(0))
c.add(gates.M(1))


result = c(nshots=3)
for i in range(2):
    symbolic_ham = Z(i)
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)

    print(f"Situa {i}")
    print("frequencies")
    print(result.frequencies())
    expectation_value = ham.expectation_from_samples(result.frequencies())
    print("expectation_value")
    print(expectation_value)

