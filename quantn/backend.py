import numpy as np
import tensornetwork as tn
from typing import List

def create_node(value: List) -> tn.Node:
    tensor = np.array(value, dtype=complex)
    return tn.Node(tensor)

def reshape_tensor(node: tn.Node, new_shape: List):
    node.set_tensor(np.reshape(node.get_tensor(), new_shape))
