import numpy as np
from tensornetwork.network_components import Node, Edge
from tensornetwork.contractors import greedy
from quantn import Qubit, XGate
from typing import Tuple

def test_pauli_xgate():
  inputs = [[1, 0], [0, 1]]
  outputs = [[0, 1], [1, 0]]
  for _in, _out in zip(inputs, outputs):
    ket = Qubit()
    ket.set_tensor(np.array(_in))
    result = XGate(ket)
    np.testing.assert_allclose(result.get_tensor(), np.array(_out))

