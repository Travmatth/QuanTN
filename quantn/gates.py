import random
import numpy as np
from tensornetwork import reachable, connect
from typing import Optional, Text, Tuple, Sequence
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

def tgate(edge: Edge) -> Edge:
    tensor = [[1, 0], [0, exp((1j * pi) / 4)]]
    tgate = Node(np.array(tensor, dtype=complex))
    tgate[1] ^ edge
    return tgate[0]

def controlled_xgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_ygate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0-1j],
        [0, 0, 0+1j, 0]
    ], dtype=complex)
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_zgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_hgate(control_edge: Edge,
                    target_edge: Edge) -> Tuple[Edge, Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1/sqrt(2), 1/sqrt(2)],
        [0, 0, 1/sqrt(2), -1/sqrt(2)]
    ], dtype=complex)
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

