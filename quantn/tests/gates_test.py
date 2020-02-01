import numpy as np
import pytest
from tensornetwork.contractors import greedy
import quantn.gates as gates
from quantn import create_qubit, contract_network
from math import sqrt, pi
from cmath import exp

def test_pauli_xgate_1():
  control = gates.xgate(create_qubit())
  edge = contract_network(control)
  reference = np.array([0+0j, 1+0j])
  np.testing.assert_allclose(edge.get_tensor(), reference)

def test_pauli_xgate_2():
  control = gates.xgate(gates.xgate(create_qubit()))
  target = contract_network(control)
  np.testing.assert_allclose(target.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_1():
  control = gates.ygate(create_qubit())
  target = contract_network(control).get_tensor()
  reference = np.array([0+0j, 0+1j])
  np.testing.assert_allclose(target, reference)

def test_pauli_ygate_2():
  control = gates.ygate(gates.xgate(create_qubit()))
  target = contract_network(control)
  reference = np.array([0-1j, 0+0j])
  np.testing.assert_allclose(target.get_tensor(), reference)

def test_pauli_zgate_1():
  control = gates.zgate(create_qubit())
  target = contract_network(control).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_pauli_zgate_2():
  control = gates.zgate(gates.xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  reference = np.array([0, -1])
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_1():
  control = gates.hgate(create_qubit())
  target = contract_network(control).get_tensor()
  reference = np.array([1, 1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_2():
  control = gates.hgate(gates.xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  reference = np.array([1, -1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_tgate_1():
  control = gates.tgate(create_qubit())
  target = contract_network(control).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_tgate_2():
  control = gates.tgate(gates.xgate(create_qubit()))
  target = contract_network(control).get_tensor()
  reference = np.array([0, exp((1j * pi) / 4)])
  np.testing.assert_allclose(target, reference)

def test_controlled_x_gate_1():
  # |00> controlled_x
  control = create_qubit()
  target = create_qubit()
  gate, _ = gates.controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_2():
  # |01> controlled_x
  control = create_qubit()
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_3():
  # |10> controlled_x
  control = gates.xgate(create_qubit())
  target = create_qubit()
  gate, _ = gates.controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_4():
  # |11> controlled_x
  control = gates.xgate(create_qubit())
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_xgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_1():
  # |00> controlled_y
  control = create_qubit()
  target = create_qubit()
  gate, _ = gates.controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_2():
  # |01> controlled_y
  control = create_qubit()
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_3():
  # |10> controlled_y
  control = gates.xgate(create_qubit())
  target = create_qubit()
  gate, _ = gates.controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 0+1j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_4():
  # |11> controlled_y
  control = gates.xgate(create_qubit())
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_ygate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0-1j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_1():
  # |00> controlled_z
  control = create_qubit()
  target = create_qubit()
  gate, _ = gates.controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_2():
  # |01> controlled_z
  control = create_qubit()
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_3():
  # |10> controlled_z
  control = gates.xgate(create_qubit())
  target = create_qubit()
  gate, _ = gates.controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_4():
  # |11> controlled_z
  control = gates.xgate(create_qubit())
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_zgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, -1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_1():
  # |00> controlled_h
  control = create_qubit()
  target = create_qubit()
  gate, _ = gates.controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_2():
  # |01> controlled_h
  control = create_qubit()
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_3():
  # |10> controlled_h
  # pudb.set_trace()
  control = gates.xgate(create_qubit())
  target = create_qubit()
  gate, _ = gates.controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0.70711+0j, 0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  control = gates.xgate(create_qubit())
  target = gates.xgate(create_qubit())
  gate, _ = gates.controlled_hgate(control, target)
  out = contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0.70711+0j, -0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)
