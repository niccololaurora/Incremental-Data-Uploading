import tensorflow as tf
import numpy as np
from qibo import set_backend, gates, Circuit, hamiltonians


def encoding_block():
    c = Circuit(10)
    for i in range(10):
        c.add(gates.RX(i, theta=0))
    return c


def variational_block():
    c = Circuit(10)
    for i in range(10):
        c.add(gates.RY(i, theta=0))
        c.add(gates.RZ(i, theta=0))
    for i in range(10 - 1):
        c.add(gates.CZ(i, i + 1))

    c.add(gates.CZ(9, 0))
    return c


c = variational_block()
print(c.draw())
