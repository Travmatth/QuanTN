from cmath import exp
from typing import Tuple, Sequence, Callable
from math import sqrt, pi
import tensornetwork as tn
import quantn.backend as backend

def apply(state: Sequence[tn.Node], gate: Callable, *qubits: Sequence[int]) -> None: 
    n = len(qubits)
    if n > 2:
        raise ValueError("Error, may only apply gates to up to two qubits")
    elif n == 2:
        q_0 = qubits[0]
        q_1 = qubits[1]
        control, target = gate(state[q_0], state[q_1])
        state[q_0], state[q_1] = control, target
        return
    q_0 = qubits[0]
    control = gate(state[q_0])
    state[q_0] = control

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

def controlled_xgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    gate = backend.create_node(tensor, with_shape=[2, 2, 2, 2])
    gate[0] ^ control_edge
    gate[3] ^ target_edge
    return gate[1], gate[2]

def controlled_ygate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0-1j], [0, 0, 0+1j, 0]]
    gate = backend.create_node(tensor, with_shape=[2, 2, 2, 2])
    gate[0] ^ control_edge
    gate[3] ^ target_edge
    return gate[1], gate[2]

def controlled_zgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    gate = backend.create_node(tensor, with_shape=[2, 2, 2, 2])
    gate[0] ^ control_edge
    gate[3] ^ target_edge
    return gate[1], gate[2]

def controlled_hgate(control_edge: tn.Edge,
                    target_edge: tn.Edge) -> Tuple[tn.Edge, tn.Edge]:
    tensor = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/sqrt(2), 1/sqrt(2)],
            [0, 0, 1/sqrt(2), -1/sqrt(2)]]
    gate = backend.create_node(tensor, with_shape=[2, 2, 2, 2])
    gate[0] ^ control_edge
    gate[3] ^ target_edge
    return gate[1], gate[2]
