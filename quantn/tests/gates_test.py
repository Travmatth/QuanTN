import numpy as np
import pytest
import quantn.gates as gates
import quantn as qu
from math import sqrt, pi
from cmath import exp

def test_pauli_xgate_1():
  qubits = [gates.xgate(qu.create_qubit())]
  edge = qu.contract_network(qubits)
  reference = np.array([0+0j, 1+0j])
  np.testing.assert_allclose(edge.get_tensor(), reference)

def test_pauli_xgate_2():
  qubits = [gates.xgate(gates.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits)
  np.testing.assert_allclose(target.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_1():
  qubits = [gates.ygate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0+0j, 0+1j])
  np.testing.assert_allclose(target, reference)

def test_pauli_ygate_2():
  qubits = [gates.ygate(gates.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits)
  reference = np.array([0-1j, 0+0j])
  np.testing.assert_allclose(target.get_tensor(), reference)

def test_pauli_zgate_1():
  qubits = [gates.zgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_pauli_zgate_2():
  qubits = [gates.zgate(gates.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0, -1])
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_1():
  qubits = [gates.hgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_hadamard_gate_2():
  qubits = [gates.hgate(gates.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, -1])/sqrt(2)
  np.testing.assert_allclose(target, reference)

def test_tgate_1():
  qubits = [gates.tgate(qu.create_qubit())]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([1, 0])
  np.testing.assert_allclose(target, reference)

def test_tgate_2():
  qubits = [gates.tgate(gates.xgate(qu.create_qubit()))]
  target = qu.contract_network(qubits).get_tensor()
  reference = np.array([0, exp((1j * pi) / 4)])
  np.testing.assert_allclose(target, reference)

def test_controlled_x_gate_1():
  # |00> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = gates.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_2():
  # |01> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_3():
  # |10> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_x_gate_4():
  # |11> controlled_x
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_xgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_1():
  # |00> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = gates.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_2():
  # |01> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_3():
  # |10> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, 0+1j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_y_gate_4():
  # |11> controlled_y
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_ygate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0-1j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_1():
  # |00> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = gates.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_2():
  # |01> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_3():
  # |10> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 1+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_z_gate_4():
  # |11> controlled_z
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_zgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [0+0j, -1+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_1():
  # |00> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0], qubits[1] = gates.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_2():
  # |01> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
  np.testing.assert_allclose(out, reference)

def test_controlled_h_gate_3():
  # |10> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0.70711+0j], [0+0j, 0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  qubits = [qu.create_qubit() for _ in range(2)]
  qubits[0] = gates.xgate(qu.create_qubit())
  qubits[1] = gates.xgate(qu.create_qubit())
  qubits[0], qubits[1] = gates.controlled_hgate(qubits[0], qubits[1])
  out = qu.contract_network(qubits).get_tensor()
  reference = np.array([[0+0j, 0.70711+0j], [0+0j, -0.70711+0j]])
  np.testing.assert_allclose(out, reference, atol=3.21881345e-06)
