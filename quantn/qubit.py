import random
import numpy as np
from typing import Optional, Text, Union, Sequence
import tensornetwork as tn
import tensornetwork.contractors as cn

def create_qubit() -> tn.Edge:
  tensor = tn.Node(np.array([1, 0], dtype=complex))
  return tensor[0]

order = Optional[Sequence[tn.Edge]]

def contract_network(edge: tn.Edge, output_edge_order: order = None) -> tn.Node:
  network = tn.reachable(edge)
  ignore_edge_order = True if output_edge_order == None else False
  return cn.greedy(network,
                  output_edge_order=output_edge_order,
                  ignore_edge_order=ignore_edge_order)

def eval_probability(node: Union[tn.Edge, tn.Node]) -> np.ndarray:
  if not isinstance(node, tn.Node):
    raise ValueError("Qubit must be of tensornetwork.Node type")
  elif len(node.get_all_nondangling()) != 0:
    raise ValueError("Tensor network must be contracted before taking \
      probability amplitude")
  state = node.get_tensor()
  state = np.abs(state) ** 2
  state /= np.sum(state)
  return state

def take_bitstring(node: Union[tn.Edge, tn.Node]) -> Text:
  if not isinstance(node, tn.Node):
    raise ValueError("Node must be of tensornetwork.Node type")
  elif len(node.get_all_nondangling()) != 0:
    raise ValueError("Tensor network must be contracted before taking \
      bitstring")
  state_vector = eval_probability(node)
  flattened_state_vector = state_vector.ravel()
  indices = [i for i in range(len(flattened_state_vector))]
  linear_index = np.random.choice(indices, p=flattened_state_vector)
  random_index = np.unravel_index(linear_index, state_vector.shape)
  bitstring = ''.join(str(bit) for bit in random_index)
  return bitstring