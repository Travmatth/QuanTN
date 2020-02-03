import random
import numpy as np
import pytest
import quantn.gates as gates
import  quantn as qu
from math import sqrt, pi
from cmath import exp
import cirq

def test_3_qubit_connections():
	# |0> -X-.---
	#        |
	# |0> ---*-.-
	#          |
	# |0> -----*-
	q_0, q_1, q_2 = qu.create_qubit(), qu.create_qubit(), qu.create_qubit()
	c0_control, c0_target = gates.controlled_xgate(gates.xgate(q_0), q_1)
	c1_control, c1_target = gates.controlled_xgate(c0_target, q_2)
	reference = [[[0+0j, 0+0j], [0+0j, 0+0j]],
				[[0+0j, 0+0j], [0+0j, 1+0j]]]
	out = qu.contract_network(c1_control).get_tensor()
	np.testing.assert_allclose(out, reference)

def test_bitstring_rejects_edges():
	q_0 = qu.create_qubit()
	with pytest.raises(ValueError):
		out = qu.take_bitstring(q_0)

def test_bitstring_rejects_uncontracted():
	q_0 = gates.xgate(qu.create_qubit())
	with pytest.raises(ValueError):
		out = qu.take_bitstring(q_0.node1)

def test_take_single_qubit_bitstring():
	q_0 = qu.create_qubit()
	contracted = qu.contract_network(q_0)
	out = qu.take_bitstring(contracted)
	assert out == '0'

def test_take_multi_qubit_bitstring():
	q_0, q_1, q_2 = qu.create_qubit(), qu.create_qubit(), qu.create_qubit()
	c0_control, c0_target = gates.controlled_xgate(gates.xgate(q_0), q_1)
	c1_control, c1_target = gates.controlled_xgate(c0_target, q_2)
	contracted = qu.contract_network(c1_control)
	out = qu.take_bitstring(contracted)
	assert out == "111"

ops = [
	("X", 1, cirq.X, gates.xgate),
	("Y", 1, cirq.Y, gates.ygate),
	("Z", 1, cirq.Z, gates.zgate),
	("H", 1, cirq.H, gates.hgate),
	("T", 1, cirq.T, gates.tgate),
	("CX", 2, cirq.CX, gates.controlled_xgate),
	("CY", 2, cirq.CY, gates.controlled_ygate),
	("CZ", 2, cirq.CZ, gates.controlled_zgate),
	("CH", 2, cirq.CH, gates.controlled_hgate),
]

"""
def test_random_circuits():
	for _ in range(5):
		num_qubits = random.randint(0, 21)
		num_gates = random.rand_int(0, 11)
		cirq_result, quantn_result = None, None
		for i in range(num_gates):
			quantn_qubits = [qu.create_qubit() for _ in range(num_qubits)]
			cirq_qubits = [cirq.GridQubit(0, j) for j in range(num_qubits)]
			op = random.randint(0, len(ops))
			q_0 = random.randint(0, num_qubits)
			if ops[op][1] == 1:
				target = ops[op][3](quantn_qubits[q_0])
				quantn_qubits[q_0] = target
				ops[op][2](cirq_qubits[q_0])
			else:
				q_1 = None
				while q_0 != q_1:
					q_1 = random.randint(0, num_qubits)
				control, target = ops[op][3](quantn_qubits[q_0],
											quantn_qubits[q_1])
				quantn_qubits[q_0] = control
				quantn_qubits[q_1] = target
"""
