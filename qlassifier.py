import tensorflow as tf
from qibo import set_backend, gates, Circuit


def encoding_block(nqubits, params):
    c = Circuit(nqubits)

    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    return c


def variational_block(nqubits):
    c = Circuit(nqubits)

    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    for i in range(int(nqubits / 2)):
        c.add(gates.CZ(i, nqubits - (i + 1)))

    return c


def circuit_nosplit():
    for i in range(nqubits):
        for j in range(nqubits):
            c.add(gates.RX(i, theta=0))

    for i in range(nqubits):
        for j in range(nqubits - 1):
            c.add(gates.CZ(i, i + 1))


def circuito_10(nqubits):
    c = Circuit(nqubits)

    # 1 encoding block
    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    # variational block
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    for i in range(int(nqubits / 2)):
        c.add(gates.CZ(i, nqubits - (i + 1)))

    # 2 encoding block
    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    # variational block
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    for i in range(int(nqubits / 2)):
        c.add(gates.CZ(i, nqubits - (i + 1)))

    # 3 encoding block
    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    # variational block
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    for i in range(int(nqubits / 2)):
        c.add(gates.CZ(i, nqubits - (i + 1)))

    # 4 encoding block
    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    # variational block
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))

    for i in range(int(nqubits / 2)):
        c.add(gates.CZ(i, nqubits - (i + 1)))

    # 5 encoding block
    for i in range(nqubits):
        c.add(gates.RX(i, theta=params[i]))

    return c
