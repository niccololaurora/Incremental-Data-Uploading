import tensorflow as tf
import numpy as np
from qibo import set_backend, gates, Circuit, hamiltonians
from qibo.optimizers import optimize, sgd, cmaes
from help_functions import batch_data, calculate_batches, label_converter


layers = 2
vparams = np.random.normal(loc=0, scale=1, size=(20 * layers, 20))

print(vparams[0])
print(vparams[1])
