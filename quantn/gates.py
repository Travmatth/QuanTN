import random
import numpy as np
from tensornetwork import reachable, connect
from typing import Optional, Text, Tuple
from quantn.qubit import create_qubit
from tensornetwork.network_components import Tensor, Node, BaseNode, Edge, \
    connect, contract, CopyNode
from tensornetwork.contractors import greedy, bucket
from math import sqrt, pi
from cmath import exp

def xgate(edge: Edge) -> Edge:
    xgate = Node(np.array([[0, 1], [1, 0]], dtype=complex))
    edge ^ xgate[1]
    return xgate[0]

def ygate(edge: Edge) -> Edge:
    ygate = Node(np.array([[0, 0-1j], [0+1j, 0]], dtype=complex))
    edge ^ ygate[1]
    return ygate[0]

def zgate(edge: Edge) -> Edge:
    zgate = Node(np.array([[1, 0], [0, -1]], dtype=complex))
    edge ^ zgate[1]
    return zgate[0]

def hgate(edge: Edge) -> Edge:
    hgate = Node(np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2))
    hgate[1] ^ edge
    return hgate[0]

def tgate(edge: Edge, hermitian_transpose: bool = False) -> Edge:
    if not hermitian_transpose:
        tensor = [[1, 0], [0, exp((1j * pi) / 4)]]
    else:
        tensor = [[1, 0], [0, exp(-(1j * pi) / 4)]]
    tgate = Node(np.array(tensor, dtype=complex))
    tgate[1] ^ edge
    return tgate[0]

def pgate(edge: Edge, hermitian_transpose: bool = False) -> Edge:
    if not hermitian_transpose:
        tensor = [[1+0j, 0+0j], [0+0j, 0+1j]]
    else:
        tensor = [[1+0j, 0+0j], [0+0j, 0-1j]]
    pgate = Node(np.array(tensor, dtype=complex))
    pgate[1] ^ edge
    return pgate[0]

def controlled_xgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    copy = CopyNode(rank=3, dimension=2)
    tensor = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
    xor = Node(np.array(tensor, dtype=complex))
    control_edge ^ copy[0]
    target_edge ^ xor[0]
    copy[1] ^ xor[1]
    return copy[2], xor[2]

def controlled_ygate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    # qubit 0
    copy = CopyNode(rank=3, dimension=2)
    control_edge ^ copy[0]
    # qubit 1
    ht_pgate = pgate(target_edge, hermitian_transpose=True)
    tensor = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
    xor = Node(np.array(tensor, dtype=complex))
    ht_pgate ^ xor[0]
    copy[2] ^ xor[2]
    pgate_edge = pgate(xor[1])
    return copy[1], pgate_edge
    # [[0.+0.j 0.+0.j]
    # [1.+0.j 0.+0.j]]
    # # qubit 0
    # copy = CopyNode(rank=3, dimension=2)
    # control_edge ^ copy[1]
    # # qubit 1
    # ht_pgate = pgate(target_edge, hermitian_transpose=True)
    # tensor = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
    # xor = Node(np.array(tensor, dtype=complex))
    # ht_pgate ^ xor[1]
    # copy[2] ^ xor[2]
    # pgate_edge = pgate(xor[0])
    # return copy[0], pgate_edge

    # [[0.+0.j 0.+0.j]
    # [1.+0.j 0.+0.j]]
    # # qubit 0
    # copy = CopyNode(rank=3, dimension=2)
    # control_edge ^ copy[2]
    # # qubit 1
    # ht_pgate = pgate(target_edge, hermitian_transpose=True)
    # tensor = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
    # xor = Node(np.array(tensor, dtype=complex))
    # ht_pgate ^ xor[2]
    # copy[0] ^ xor[0]
    # pgate_edge = pgate(xor[1])
    # return copy[1], pgate_edge

def controlled_zgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    # qubit 0
    copy = CopyNode(rank=3, dimension=2)
    control_edge ^ copy[0]
    # qubit 1
    h1gate_edge = hgate(target_edge)
    tensor = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=complex)
    xor = Node(tensor)
    h1gate_edge ^ xor[0]
    copy[1] ^ xor[1]
    h2gate_edge = hgate(xor[2])
    return copy[2], h2gate_edge

def controlled_hgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    #qubit 0
    copy = CopyNode(rank=3, dimension=2)
    control_edge ^ copy[0]
    # qubit 1
    pgate_edge = pgate(target_edge)
    h1gate_edge = hgate(pgate_edge)
    tgate_edge = tgate(h1gate_edge)
    xor = Node(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]))
    tgate_edge ^ xor[0]
    copy[1] ^ xor[1]
    ht_tgate_edge = tgate(xor[2], hermitian_transpose=True)
    h2gate_edge = hgate(ht_tgate_edge)
    ht_pgate_edge = pgate(h2gate_edge, hermitian_transpose=True)
    return copy[2], ht_pgate_edge
