import numpy as np
from quantn import create_qubit, contract_network

def test_qubit_init():
  q_0_edge = create_qubit()
  assert q_0_edge.node1.shape[0] == 2

def test_contract_to_scalar():
  q_0_edge = create_qubit()
  q_1_edge = create_qubit()
  q_0_edge ^ q_1_edge
  out = contract_network(q_0_edge)
  assert out.get_tensor() == 1
