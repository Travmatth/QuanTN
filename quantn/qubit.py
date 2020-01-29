import random
import numpy as np
import tensornetwork
from typing import Optional, Text
from tensornetwork.network_components import Node

class Qubit(Node):
  def __init__(self, name: Optional[Text] = None, axis: Optional[Text] = None):
    super().__init__(tensor=np.array([1, 0]), name=name, axis_names=axis)

  def bitstring(self) -> Text:
    state = super().get_tensor()
    alpha, beta = state[0], state[1]
    weights = [abs(alpha)**2, abs(beta)**2]
    strings = random.choices(['0', '1'], weights=weights)
    return strings[0]

