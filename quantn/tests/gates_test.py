import numpy as np
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

def test_pauli_xgate_zero_ket():
  edge = xgate(create_qubit())
  out = contract_network(edge)
  np.testing.assert_allclose(out.get_tensor(), np.array([0+0j, 1+0j]))

def test_pauli_xgate_one_ket():
  edge = xgate(xgate(create_qubit()))
  out = contract_network(edge)
  np.testing.assert_allclose(out.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_zero_ket():
  edge = ygate(create_qubit())
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([0+0j, 0+1j], dtype=complex))
  
def test_pauli_ygate_one_ket():
  edge = ygate(xgate(create_qubit()))
  out = contract_network(edge)
  np.testing.assert_allclose(out.get_tensor(), np.array([0-1j, 0+0j]))
  
def test_pauli_zgate():
  edge = zgate(create_qubit())
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([1, 0]))
  edge = zgate(xgate(create_qubit()))
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([0, -1]))

def test_hadamard_gate():
  edge = hgate(create_qubit())
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([1, 1])/sqrt(2))
  edge = hgate(xgate(create_qubit()))
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([1, -1])/sqrt(2))

def test_tgate():
  edge = tgate(create_qubit())
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([1, 0]))
  edge = tgate(xgate(create_qubit()))
  out = contract_network(edge).get_tensor()
  np.testing.assert_allclose(out, np.array([0, exp((1j * pi) / 4)]))

@flaky
def test_controlled_x_gate():
  # |00> x controlled_x -> |00>
  control_edge = create_qubit()
  target_edge = create_qubit()
  copy_edge, xor_edge = controlled_xgate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

  # |01> x controlled_x -> |01>
  control_edge = create_qubit()
  target_edge = xgate(create_qubit())
  copy_edge, xor_edge = controlled_xgate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

  # |10> x controlled_x -> |11>
  control_edge = xgate(create_qubit())
  target_edge = create_qubit()
  copy_edge, xor_edge = controlled_xgate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 1+0j]]))

  # |11> x controlled_x -> |10>
  control_edge = xgate(create_qubit())
  target_edge = xgate(create_qubit())
  copy_edge, xor_edge = controlled_xgate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [1+0j, 0+0j]]))

def test_controlled_y_gate():
  # |00> x controlled_y -> |00>
  control_edge = create_qubit()
  target_edge = create_qubit()
  copy_edge, xor_edge = controlled_ygate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))
  # |01> x controlled_y -> |01>
  control_edge = create_qubit()
  target_edge = xgate(create_qubit())
  copy_edge, xor_edge = controlled_ygate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))
  def ref_y(a: Edge, b: Edge) -> Tuple[Edge, Edge]:
    # outer = outer_product(a.node1, b.node1)
    tensor = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0-1j], [0, 0, 0+1j, 0]]
    cy = Node(np.array(np.reshape(tensor, [2, 2, 2, 2]), dtype=complex))
    # cy[1] ^ outer[0]
    cy[1] ^ a
    cy[2] ^ b
    return cy[0]

  # |10> x controlled_y -> |11>
  control_edge = xgate(create_qubit())
  target_edge = create_qubit()
  copy_edge, xor_edge = controlled_ygate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 0+1j]]))
  # print(control_edge.node1.get_tensor(), target_edge.get_tensor(), out)
#   # assert True == False

  # |11> x controlled_y -> |10>
  # import pudb
  # pudb.set_trace()
  # _control_edge = xgate(create_qubit())
  # _target_edge = create_qubit()
  # ref = ref_y(_control_edge, _target_edge)
  # out = contract_network(ref).get_tensor()
  # #
  control_edge = xgate(create_qubit())
  target_edge = xgate(create_qubit())
  copy_edge, xor_edge = controlled_xgate(control_edge, target_edge)
  out = contract_network(copy_edge).get_tensor()
  print(out)
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0-1j, 0+0j]]))
