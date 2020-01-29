import random
import numpy as np
import tensornetwork
from typing import Optional, Text, Union
from tensornetwork.network_components import Node, BaseNode, Tensor

class Qubit(Node):
  def __init__(self,
              tensor: Optional[Union[Tensor, BaseNode]] = None,
              name: Optional[Text] = None):
    tensor = np.array([1, 0]) if tensor is None else tensor
    super().__init__(tensor=tensor, name=name, axis_names=["edge"])

  def bitstring(self) -> Text:
    state = super().get_tensor()
    alpha, beta = state[0], state[1]
    weights = [abs(alpha)**2, abs(beta)**2]
    strings = random.choices(['0', '1'], weights=weights)
    return strings[0]

