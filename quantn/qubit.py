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
  if isinstance(node, tn.Edge):
    raise ValueError("Qubit must be of tensornetwork.Node type")
  elif len(node.get_all_nondangling()) != 0:
    raise ValueError("Tensor network must be contracted before taking \
      probability amplitude")
  return node.get_tensor()

def take_bitstring(node: Union[tn.Edge, tn.Node]) -> Text:
  if isinstance(node, tn.Edge):
    raise ValueError("Node must be of tensornetwork.Node type")
  elif node.get_rank() == 1:
    alpha, beta = node.get_tensor()
    weights = [abs(alpha)**2, abs(beta)**2]
    return random.choices(['0', '1'], weights=weights)