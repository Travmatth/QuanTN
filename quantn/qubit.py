import random
import numpy as np
from typing import Optional, Text, Union, Sequence, Tuple
import tensornetwork as tn
import tensornetwork.contractors as cn

def create_qubit() -> tn.Edge:
  tensor = tn.Node(np.array([1, 0], dtype=complex))
  return tensor[0]

def contract_network(edges: Sequence[tn.Edge]) -> tn.Node:
  network = set()    
  for edge in edges:
    network |= tn.reachable(edge)
  return cn.greedy(network, output_edge_order=edges)

def eval_probability(node: tn.Node, normalize: bool = False) -> np.ndarray:
  if not isinstance(node, tn.Node):
    raise ValueError("Qubit must be of tensornetwork.Node type")
  elif len(node.get_all_nondangling()) != 0:
    raise ValueError("Tensor network must be contracted before taking \
      probability amplitude")
  state = node.get_tensor()
  if normalize:
    state = np.abs(state) ** 2
    state /= np.sum(state)
  return state.ravel()

def take_bitstring(node: Union[tn.Edge, tn.Node]) -> Text:
  if not isinstance(node, tn.Node):
    raise ValueError("Node must be of tensornetwork.Node type")
  elif len(node.get_all_nondangling()) != 0:
    raise ValueError("Tensor network must be contracted before taking \
      bitstring")
  state_vector = eval_probability(node, normalize=True)
  indices = [i for i in range(len(state_vector))]
  linear_index = np.random.choice(indices, p=state_vector)
  random_index = np.unravel_index(linear_index, node.shape)
  bitstring = ''.join(str(bit) for bit in random_index)
  return bitstring
