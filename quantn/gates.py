import numpy as np
from typing import Optional, Text, Tuple, Sequence
from quantn.qubit import create_qubit
import quantn.backend as backend
import tensornetwork as tn
from math import sqrt, pi
from cmath import exp

def xgate(edge: tn.Edge) -> tn.Edge:
    gate = backend.create_node([[0, 1], [1, 0]])
    edge ^ gate[1]
    return gate[0]

def ygate(edge: tn.Edge) -> tn.Edge:
    gate = backend.create_node([[0, 0-1j], [0+1j, 0]])
    edge ^ gate[1]
    return gate[0]

def zgate(edge: tn.Edge) -> tn.Edge:
    gate = backend.create_node([[1, 0], [0, -1]])
    edge ^ gate[1]
    return gate[0]

def hgate(edge: tn.Edge) -> tn.Edge:
    gate = backend.create_node([[1/sqrt(2), 1/sqrt(2)],
                                [1/sqrt(2), -1/sqrt(2)]])
    edge ^ gate[1]
    return gate[0]

def tgate(edge: tn.Edge) -> tn.Edge:
    gate = backend.create_node([[1, 0], [0, exp((1j * pi) / 4)]])
    edge ^ gate[1]
    return gate[0]

import numpy as np
def controlled_xgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    gate = backend.create_node([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]])
    tensor = np.array()
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = tn.Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_ygate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0-1j],
        [0, 0, 0+1j, 0]
    ])
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = tn.Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_zgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = tn.Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]

def controlled_hgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1/sqrt(2), 1/sqrt(2)],
        [0, 0, 1/sqrt(2), -1/sqrt(2)]
    ])
    tensor = np.reshape(tensor, [2, 2, 2, 2])
    ch = tn.Node(tensor)
    ch[2] ^ control_edge
    ch[3] ^ target_edge
    return ch[0], ch[1]
