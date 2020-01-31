import random
import numpy as np
from typing import Optional, Text, Union, Sequence, Callable
from tensornetwork import reachable
from tensornetwork.network_components import Edge, Node, BaseNode, Tensor
from tensornetwork.contractors import auto

def create_qubit() -> Edge:
  tensor = Node(np.array([1, 0], dtype=complex))
  return tensor[0]

def contract_network(edge: Edge,
                    output_edge_order: Optional[Sequence[Edge]] = None,
                    strategy: Callable = auto) -> Node:
  network = reachable(edge)
  ignore_edge_order = True if output_edge_order == None else False
  return strategy(network,
                  output_edge_order=output_edge_order,
                  ignore_edge_order=ignore_edge_order)

