import numpy as np
import pytest
from tensornetwork.network_components import Node, Edge, outer_product
from tensornetwork.contractors import greedy
from tensornetwork import reachable
from quantn import create_qubit, contract_network, \
                  xgate, ygate, zgate, hgate, tgate, \
                  controlled_xgate, controlled_ygate, \
                  controlled_zgate, controlled_hgate
from typing import Tuple
from math import sqrt, pi
from cmath import exp
from flaky import flaky

def test_pauli_xgate_1():
  control = xgate(create_qubit())
  edge = contract_network(control)
  np.testing.assert_allclose(edge.get_tensor(), np.array([0+0j, 1+0j]))

def test_pauli_xgate_2():
  control = xgate(xgate(create_qubit()))
  target = contract_network(control)
  np.testing.assert_allclose(target.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_1():
  control = ygate(create_qubit())
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([0+0j, 0+1j], dtype=complex))
  
def test_pauli_ygate_2():
  control = ygate(xgate(create_qubit()))
  target = contract_network(control)
  np.testing.assert_allclose(target.get_tensor(), np.array([0-1j, 0+0j]))
  
def test_pauli_zgate_1():
  control = zgate(create_qubit())
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([1, 0]))

def test_pauli_zgate_2():
  control = zgate(xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([0, -1]))

def test_hadamard_gate_1():
  control = hgate(create_qubit())
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([1, 1])/sqrt(2))

def test_hadamard_gate_2():
  control = hgate(xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([1, -1])/sqrt(2))

def test_tgate_1():
  control = tgate(create_qubit())
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([1, 0]))

def test_tgate_2():
  control = tgate(xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  np.testing.assert_allclose(target, np.array([0, exp((1j * pi) / 4)]))

def test_controlled_x_gate_1():
  # |00> controlled_x
  control = create_qubit()
  target = create_qubit()
  gate, _ = controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_x_gate_2():
  # |01> controlled_x
  control = create_qubit()
  target = xgate(create_qubit())
  gate, _ = controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_x_gate_3():
  # |10> controlled_x
  control = xgate(create_qubit())
  target = create_qubit()
  gate, _ = controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 1+0j]]))

def test_controlled_x_gate_4():
  # |11> controlled_x
  control = xgate(create_qubit())
  target = xgate(create_qubit())
  gate, _ = controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [1+0j, 0+0j]]))

def test_controlled_y_gate_1():
  # |00> controlled_y
  control = create_qubit()
  target = create_qubit()
  gate, _ = controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_y_gate_2():
  # |01> controlled_y
  control = create_qubit()
  target = xgate(create_qubit())
  gate, _ = controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_y_gate_3():
  # |10> controlled_y
  control = xgate(create_qubit())
  target = create_qubit()
  gate, _ = controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 0+1j]]))

def test_controlled_y_gate_4():
  # |11> controlled_y
  control = xgate(create_qubit())
  target = xgate(create_qubit())
  gate, _ = controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0-1j, 0+0j]]))

def test_controlled_z_gate_1():
  # |00> controlled_z
  control = create_qubit()
  target = create_qubit()
  gate, _ = controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_z_gate_2():
  # |01> controlled_z
  control = create_qubit()
  target = xgate(create_qubit())
  gate, _ = controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_z_gate_3():
  # |10> controlled_z
  control = xgate(create_qubit())
  target = create_qubit()
  gate, _ = controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [1+0j, 0+0j]]))

def test_controlled_z_gate_4():
  # |11> controlled_z
  control = xgate(create_qubit())
  target = xgate(create_qubit())
  gate, _ = controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, -1+0j]]))

def test_controlled_h_gate_1():
  # |00> controlled_h
  control = create_qubit()
  target = create_qubit()
  gate, _ = controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j], [0+0j, 0+0j]]))

def test_controlled_h_gate_2():
  # |01> controlled_h
  control = create_qubit()
  target = xgate(create_qubit())
  gate, _ = controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j], [0+0j, 0+0j]]))

def test_controlled_h_gate_3():
  # |10> controlled_h
  # pudb.set_trace()
  control = xgate(create_qubit())
  target = create_qubit()
  gate, _ = controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                            [0.70711+0j, 0.70711+0j]]), atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  control = xgate(create_qubit())
  target = xgate(create_qubit())
  gate, _ = controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                             [0.70711+0j, -0.70711+0j]]), atol=3.21881345e-06)
