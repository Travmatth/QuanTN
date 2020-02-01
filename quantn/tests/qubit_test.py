import pytest
import numpy as np
from quantn import create_qubit, contract_network, eval_probability, xgate

def test_qubit_init():
  q_0_edge = create_qubit()
  assert q_0_edge.node1.shape[0] == 2

def test_contract_to_scalar():
  q_0_edge = create_qubit()
  q_1_edge = create_qubit()
  q_0_edge ^ q_1_edge
  out = contract_network(q_0_edge)
  assert out.get_tensor() == 1

def test_eval_probability_rejects_edges():
  q_0 = create_qubit()
  with pytest.raises(ValueError):
    state = eval_probability(q_0)

def test_eval_probability_rejects_uncontracted():
  q_0 = create_qubit()
  gate = xgate(q_0)
  with pytest.raises(ValueError):
    state = eval_probability(q_0)

def test_eval_probability_returns_probability_amplitude():
  q_0 = create_qubit()
  reference = np.array([1, 0])
  state = eval_probability(q_0.node1)
  np.testing.assert_allclose(state, reference)
