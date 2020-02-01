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

def test_pauli_xgate_zero_ket():
  control = xgate(create_qubit())
  edge = contract_network(control)
  np.testing.assert_allclose(edge.get_tensor(), np.array([0+0j, 1+0j]))

def test_pauli_xgate_one_ket():
  b = xgate(xgate(create_qubit()))
  d = contract_network(b)
  np.testing.assert_allclose(d.get_tensor(), np.array([1+0j, 0+0j]))

def test_pauli_ygate_zero_ket():
  e = ygate(create_qubit())
  f = contract_network(e).get_tensor()
  np.testing.assert_allclose(f, np.array([0+0j, 0+1j], dtype=complex))
  
def test_pauli_ygate_one_ket():
  g = ygate(xgate(create_qubit()))
  h = contract_network(g)
  np.testing.assert_allclose(h.get_tensor(), np.array([0-1j, 0+0j]))
  
def test_pauli_zgate():
  i = zgate(create_qubit())
  j = contract_network(i).get_tensor()
  np.testing.assert_allclose(j, np.array([1, 0]))
  i = zgate(xgate(create_qubit()))
  j = contract_network(i).get_tensor()
  np.testing.assert_allclose(j, np.array([0, -1]))

def test_hadamard_gate():
  k = hgate(create_qubit())
  l = contract_network(k).get_tensor()
  np.testing.assert_allclose(l, np.array([1, 1])/sqrt(2))
  k = hgate(xgate(create_qubit()))
  l = contract_network(k).get_tensor()
  np.testing.assert_allclose(l, np.array([1, -1])/sqrt(2))

def test_tgate():
  m = tgate(create_qubit())
  n = contract_network(m).get_tensor()
  np.testing.assert_allclose(n, np.array([1, 0]))
  m = tgate(xgate(create_qubit()))
  n = contract_network(m).get_tensor()
  np.testing.assert_allclose(n, np.array([0, exp((1j * pi) / 4)]))

def test_controlled_x_gate_1():
  # |00> controlled_x
  o = create_qubit()
  p = create_qubit()
  q, _ = controlled_xgate(o, p)
  out = contract_network(q).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_x_gate_2():
  # |01> controlled_x
  r = create_qubit()
  s = xgate(create_qubit())
  t, _ = controlled_xgate(r, s)
  out = contract_network(t).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_x_gate_3():
  # |10> controlled_x
  u = xgate(create_qubit())
  v = create_qubit()
  w, _ = controlled_xgate(u, v)
  out = contract_network(w).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 1+0j]]))

@flaky
def test_controlled_x_gate_4():
  # |11> controlled_x
  x = xgate(create_qubit())
  y = xgate(create_qubit())
  z, _ = controlled_xgate(x, y)
  out = contract_network(z).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [1+0j, 0+0j]]))

def test_controlled_y_gate_1():
  # |00> controlled_y
  aa = create_qubit()
  bb = create_qubit()
  cc, _ = controlled_ygate(aa, bb)
  out = contract_network(cc).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_y_gate_2():
  # |01> controlled_y
  dd = create_qubit()
  ee = xgate(create_qubit())
  ff, _ = controlled_ygate(dd, ee)
  out = contract_network(ff).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_y_gate_3():
  # |10> controlled_y
  gg = xgate(create_qubit())
  hh = create_qubit()
  ii, _ = controlled_ygate(gg, hh)
  out = contract_network(ii).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, 0+1j]]))

def test_controlled_y_gate_4():
  # |11> controlled_y
  jj = xgate(create_qubit())
  kk = xgate(create_qubit())
  ll, _ = controlled_ygate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0-1j, 0+0j]]))

def test_controlled_z_gate_1():
  # |00> controlled_z
  jj = create_qubit()
  kk = create_qubit()
  ll, _ = controlled_zgate(jj, kk)
  out = contract_network(ll).get_tensor()
  print(out - np.array([[1+0j, 0+0j], [0+0j, 0+0j]]))
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_z_gate_2():
  # |01> controlled_z
  jj = create_qubit()
  kk = xgate(create_qubit())
  ll, _ = controlled_zgate(jj, kk)
  out = contract_network(ll).get_tensor()
  print(out - np.array([[1+0j, 0+0j], [0+0j, 0+0j]]))
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j],
                                            [0+0j, 0+0j]]))

def test_controlled_z_gate_3():
  # |10> controlled_z
  jj = xgate(create_qubit())
  kk = create_qubit()
  ll, _ = controlled_zgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [1+0j, 0+0j]]))

def test_controlled_z_gate_4():
  # |11> controlled_z
  jj = xgate(create_qubit())
  kk = xgate(create_qubit())
  ll, _ = controlled_zgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j],
                                            [0+0j, -1+0j]]))

def test_controlled_h_gate_1():
  # |00> controlled_h
  jj = create_qubit()
  kk = create_qubit()
  ll, _ = controlled_hgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[1+0j, 0+0j], [0+0j, 0+0j]]))

def test_controlled_h_gate_2():
  # |01> controlled_h
  jj = create_qubit()
  kk = xgate(create_qubit())
  ll, _ = controlled_hgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 1+0j], [0+0j, 0+0j]]))

def test_controlled_h_gate_3():
  # |10> controlled_h
  # pudb.set_trace()
  jj = xgate(create_qubit())
  kk = create_qubit()
  ll, _ = controlled_hgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j], [0.70711+0j, 0.70711+0j]]), atol=3.21881345e-06)

def test_controlled_h_gate_4():
  # |11> controlled_h
  jj = xgate(create_qubit())
  kk = xgate(create_qubit())
  ll, _ = controlled_hgate(jj, kk)
  out = contract_network(ll).get_tensor()
  np.testing.assert_allclose(out, np.array([[0+0j, 0+0j], [0.70711+0j, -0.70711+0j]]), atol=3.21881345e-06)
