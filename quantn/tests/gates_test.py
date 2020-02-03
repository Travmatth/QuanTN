import numpy as np
import pytest
import quantn.gates as gates
import quantn as qu
from math import sqrt, pi
from cmath import exp

def test_pauli_xgate_1():
  control = gates.xgate(qu.create_qubit())
  edge = qu.contract_network(control)
  reference = np.array([0+0j, 1+0j])
  np.testing.assert_allclose(edge.get_tensor(), reference)

def test_pauli_xgate_2():
  control = gates.xgate(gates.xgate(qu.create_qubit()))
  target = qu.contract_network(control)
  np.testing.assert_allclose(target.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_1():
  control = gates.ygate(qu.create_qubit())
  target = qu.contract_network(control).get_tensor()
  reference = np.array([0+0j, 0+1j])
  np.testing.assert_allclose(target, reference)

def test_pauli_ygate_2():
  control = gates.ygate(gates.xgate(qu.create_qubit()))
  target = qu.contract_network(control)
  reference = np.array([0-1j, 0+0j])
  np.testing.assert_allclose(target.get_tensor(), reference)

def test_pauli_zgate_1():
  control = gates.zgate(qu.create_qubit())
  target = qu.contract_network(control).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_pauli_zgate_2():
  control = gates.zgate(gates.xgate(qu.create_qubit()))
  target = qu.contract_network(control).get_tensor()
  reference = np.array([0, -1])
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_1():
  control = gates.hgate(qu.create_qubit())
  target = qu.contract_network(control).get_tensor()
  reference = np.array([1, 1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_2():
  control = gates.hgate(gates.xgate(qu.create_qubit()))
  target = qu.contract_network(control).get_tensor()
  reference = np.array([1, -1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_tgate_1():
  control = gates.tgate(qu.create_qubit())
  target = qu.contract_network(control).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_tgate_2():
  control = gates.tgate(gates.xgate(qu.create_qubit()))
  target = qu.contract_network(control).get_tensor()
  reference = np.array([0, exp((1j * pi) / 4)])
  np.testing.assert_allclose(target, reference)

def test_controlled_x_gate_1():
  # |00> controlled_x
  control = qu.create_qubit()
  target = qu.create_qubit()
  gate, _ = gates.controlled_xgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_2():
  # |01> controlled_x
  control = qu.create_qubit()
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_xgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_3():
  # |10> controlled_x
  control = gates.xgate(qu.create_qubit())
  target = qu.create_qubit()
  gate, _ = gates.controlled_xgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_4():
  # |11> controlled_x
  control = gates.xgate(qu.create_qubit())
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_xgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_1():
  # |00> controlled_y
  control = qu.create_qubit()
  target = qu.create_qubit()
  gate, _ = gates.controlled_ygate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_2():
  # |01> controlled_y
  control = qu.create_qubit()
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_ygate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_3():
  # |10> controlled_y
  control = gates.xgate(qu.create_qubit())
  target = qu.create_qubit()
  gate, _ = gates.controlled_ygate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 0+1j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_4():
  # |11> controlled_y
  control = gates.xgate(qu.create_qubit())
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_ygate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0-1j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_1():
  # |00> controlled_z
  control = qu.create_qubit()
  target = qu.create_qubit()
  gate, _ = gates.controlled_zgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_2():
  # |01> controlled_z
  control = qu.create_qubit()
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_zgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_3():
  # |10> controlled_z
  control = gates.xgate(qu.create_qubit())
  target = qu.create_qubit()
  gate, _ = gates.controlled_zgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_4():
  # |11> controlled_z
  control = gates.xgate(qu.create_qubit())
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_zgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, -1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_1():
  # |00> controlled_h
  control = qu.create_qubit()
  target = qu.create_qubit()
  gate, _ = gates.controlled_hgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_2():
  # |01> controlled_h
  control = qu.create_qubit()
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_hgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_3():
  # |10> controlled_h
  # pudb.set_trace()
  control = gates.xgate(qu.create_qubit())
  target = qu.create_qubit()
  gate, _ = gates.controlled_hgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0.70711+0j, 0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  control = gates.xgate(qu.create_qubit())
  target = gates.xgate(qu.create_qubit())
  gate, _ = gates.controlled_hgate(control, target)
  out = qu.contract_network(gate).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0.70711+0j, -0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)
