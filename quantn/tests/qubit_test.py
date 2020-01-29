import numpy as np
from quantn import Qubit

def test_qubit_init():
  q_0 = Qubit()
  assert q_0.shape[0] == 2

def test_qubit_0ket_bitstring():
  q_0 = Qubit()
  for i in range(1000):
    assert q_0.bitstring() == "0"

def test_qubit_1ket_bitstring():
  q_0 = Qubit()
  q_0.set_tensor(np.array([0, 1]))
  for i in range(1000):
    assert q_0.bitstring() == "1"

