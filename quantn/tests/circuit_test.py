import random
import numpy as np
import pytest
import quantn.gates as gates
import  quantn as qu
from math import sqrt, pi
from cmath import exp
import cirq
import tensornetwork as tn

def test_3_qubit_connections():
	# |0> -X-.---
	#        |
	# |0> ---*-.-
	#          |
	# |0> -----*-
	qubits = [qu.create_qubit() for _ in range(3)]
	qubits[0], qubits[1] = gates.controlled_xgate(gates.xgate(qubits[0]), qubits[1])
	qubits[1], qubits[2] = gates.controlled_xgate(qubits[1], qubits[2])
	reference = [[[0+0j, 0+0j], [0+0j, 0+0j]],
				[[0+0j, 0+0j], [0+0j, 1+0j]]]
	out = qu.contract_network(qubits).get_tensor()
	np.testing.assert_allclose(out, reference)

def test_bitstring_rejects_edges():
	qubits = [qu.create_qubit()]
	with pytest.raises(ValueError):
		out = qu.take_bitstring(qubits)

def test_bitstring_rejects_uncontracted():
	qubits = [gates.xgate(qu.create_qubit())]
	with pytest.raises(ValueError):
		out = qu.take_bitstring(qubits[0].node1)

def test_take_single_qubit_bitstring():
	qubits = [gates.xgate(qu.create_qubit())]
	contracted = qu.contract_network(qubits)
	out = qu.take_bitstring(contracted)
	assert out == '1'

def test_take_multi_qubit_bitstring():
	qubits = [qu.create_qubit() for _ in range(3)]
	qubits[0], qubits[1] = gates.controlled_xgate(gates.xgate(qubits[0]), qubits[1])
	qubits[1], qubits[2] = gates.controlled_xgate(qubits[1], qubits[2])
	contracted = qu.contract_network(qubits)
	out = qu.take_bitstring(contracted)
	assert out == "111"

ops = [
	("X", 1, cirq.X, gates.xgate),
	("Y", 1, cirq.Y, gates.ygate),
	("Z", 1, cirq.Z, gates.zgate),
	("H", 1, cirq.H, gates.hgate),
	("T", 1, cirq.T, gates.tgate),
	("CX", 2, cirq.ControlledGate(cirq.X), gates.controlled_xgate),
	("CY", 2, cirq.ControlledGate(cirq.Y), gates.controlled_ygate),
	("CZ", 2, cirq.ControlledGate(cirq.Z), gates.controlled_zgate),
	("CH", 2, cirq.ControlledGate(cirq.H), gates.controlled_hgate),
]

class fullprint:
    'context manager for printing full numpy arrays'

    def __init__(self, **kwargs):
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)

def test_random_2():
	# import pudb
	# pudb.set_trace()
	qubits = [qu.create_qubit() for _ in range(3)]
	qubits[2] = gates.ygate(qubits[2])
	c, t = gates.controlled_ygate(qubits[2], qubits[0])
	qubits[2], qubits[0] = c, t
	out = qu.contract_network(qubits)
	result = qu.eval_probability(out).ravel()
	ref = np.array([0+0j, 0+0j, 0+0j, 0+0j, 0+0j, -1+0j, 0+0j, 0+0j])
	np.testing.assert_allclose(result, ref)

def test_random_circuits():
	# import pudb
	# pudb.set_trace()
	simulator = cirq.Simulator()
	for _ in range(1):
		ref = ""
		cirq_circuit = cirq.Circuit()
		num_qubits = random.randint(1, 6)
		num_gates = random.randint(0, 6)
		ref += "random circuit with " + str(num_qubits) + \
				" qubits and " + str(num_gates) + " gates"
		# qubits for each library
		quantn_qubits = [qu.create_qubit() for _ in range(num_qubits)]
		cirq_qubits = [cirq.GridQubit(0, j) for j in range(num_qubits)]
		# assemble random operations
		for i in range(num_gates):
			op = random.randint(0, len(ops) - 1)
			q_0 = random.randint(0, num_qubits - 1)
			# single qubit operation
			if ops[op][1] == 1:
				ref += "\nattaching gate: " + ops[op][0] + " to qubit " + str(q_0)
				target = ops[op][3](quantn_qubits[q_0])
				quantn_qubits[q_0] = target
				cirq_circuit.append(ops[op][2](cirq_qubits[q_0]))
			# two qubit operation
			else:
				q_1 = None
				while q_1 == None or q_0 == q_1:
					q_1 = random.randint(0, num_qubits - 1)
				ref += "\nattaching gate: " + ops[op][0] + " to qubits " + str(q_0) + str(q_1)
				control, target = ops[op][3](quantn_qubits[q_0],quantn_qubits[q_1])
				quantn_qubits[q_0] = control
				quantn_qubits[q_1] = target
				cirq_circuit.append(ops[op][2](cirq_qubits[q_0], cirq_qubits[q_1]))
		# assemble cirq circuit
		cirq_result = simulator.simulate(cirq_circuit, qubit_order=cirq_qubits)
		# execute quantn circuit
		quantn_circuit = qu.contract_network(quantn_qubits)
		quantn_result = qu.eval_probability(quantn_circuit)
		# compare
		if not np.allclose(quantn_result, cirq_result.final_state):
			with fullprint():
				print(ref, "\nmine:\n", quantn_result, "ref:\n", cirq_result.final_state)
		np.testing.assert_allclose(quantn_result, cirq_result.final_state)
