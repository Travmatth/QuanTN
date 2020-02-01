import random
import numpy as np
from typing import Optional, Text, Union, Sequence, Callable
import tensornetwork as tn
from tensornetwork.network_components import Edge, Node, BaseNode, Tensor
import tensornetwork.contractors as cn

def create_qubit() -> Edge:
  tensor = Node(np.array([1, 0], dtype=complex))
  return tensor[0]

def contract_network(edge: Edge,
                    output_edge_order: Optional[Sequence[Edge]] = None) -> Node:
  network = tn.reachable(edge)
  ignore_edge_order = True if output_edge_order == None else False
  return cn.greedy(network,
                  output_edge_order=output_edge_order,
                  ignore_edge_order=ignore_edge_order)

