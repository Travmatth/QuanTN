import pytest
import numpy as np
import quantn as qu
import quantn.gates as gates

def test_qubit_init():
  q_0_edge = qu.create_qubit()
  assert q_0_edge.node1.shape[0] == 2

def test_eval_probability_rejects_edges():
  q_0 = qu.create_qubit()
  with pytest.raises(ValueError):
    state = qu.eval_probability(q_0)

def test_eval_probability_rejects_uncontracted():
  q_0 = qu.create_qubit()
  gate = gates.xgate(q_0)
  with pytest.raises(ValueError):
    state = qu.eval_probability(q_0)

def test_eval_probability_returns_probability_amplitude():
  q_0 = qu.create_qubit()
  reference = np.array([1, 0])
  state = qu.eval_probability(q_0.node1)
  np.testing.assert_allclose(state, reference)
