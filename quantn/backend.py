import numpy as np
import tensornetwork as tn
from typing import List, Optional

def create_node(value: List, with_shape: Optional[List] = None) -> tn.Node:
    if with_shape is not None:
        value = np.reshape(value, with_shape)
    tensor = np.array(value, dtype=complex)
    return tn.Node(tensor)
