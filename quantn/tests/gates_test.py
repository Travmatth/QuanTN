import numpy as np
import pytest
import quantn as qu
from math import sqrt, pi
from cmath import exp

def test_pauli_xgate_1():
  qubits = [qu.xgate(qu.create_qubit())]
  edge = qu.contract_network(qubits)
  reference = np.array([0+0j, 1+0j])
  np.testing.assert_allclose(edge.get_tensor(), reference)

def test_pauli_xgate_2():
  qubits = [qu.xgate(qu.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits)
  np.testing.assert_allclose(target.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_1():
  qubits = [qu.ygate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0+0j, 0+1j])
  np.testing.assert_allclose(target, reference)

def test_pauli_ygate_2():
  qubits = [qu.ygate(qu.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits)
  reference = np.array([0-1j, 0+0j])
  np.testing.assert_allclose(target.get_tensor(), reference)

def test_pauli_zgate_1():
  qubits = [qu.zgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_pauli_zgate_2():
  qubits = [qu.zgate(qu.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0, -1])
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_1():
  qubits = [qu.hgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_2():
  qubits = [qu.hgate(qu.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, -1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_tgate_1():
  qubits = [qu.tgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_tgate_2():
  qubits = [qu.tgate(qu.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0, exp((1j * pi) / 4)])
  np.testing.assert_allclose(target, reference)

def test_controlled_x_gate_1():
  # |00> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = qu.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_2():
  # |01> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_3():
  # |10> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_4():
  # |11> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_1():
  # |00> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = qu.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_2():
  # |01> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_3():
  # |10> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 0+1j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_4():
  # |11> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0-1j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_1():
  # |00> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = qu.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_2():
  # |01> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_3():
  # |10> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_4():
  # |11> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, -1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_1():
  # |00> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = qu.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_2():
  # |01> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_3():
  # |10> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0.70711+0j], [0+0j, 0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = qu.xgate(qu.create_qubit())
  qubits[1] = qu.xgate(qu.create_qubit())
  qubits[0], qubits[1] = qu.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0.70711+0j], [0+0j, -0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)

def test_single_apply_gate():
  qubits = [qu.create_qubit()]
  qu.apply_gate(qubits, qu.xgate, 0)
  edge = qu.contract_network(qubits)
  reference = np.array([0+0j, 1+0j])
  np.testing.assert_allclose(edge.get_tensor(), reference)

def test_multi_apply_gate():
  # |00> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qu.apply_gate(qubits, qu.controlled_xgate, 0, 1)
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)